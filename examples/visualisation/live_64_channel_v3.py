#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 64-channel plotting (16x4) using PyQtGraph, but data comes from LibEMG shared memory
via your EmagerV3Streamer (buffer_write via vstack prepend) rather than reading serial here.

IMPORTANT:
- Your streamer writes NEW data by PREPENDING (vstack) into the shared buffer.
- Therefore the shared memory array is NOT a classical ring. The newest rows are at the TOP.
- Counts (tag_count) still increase by number of rows written.
- This consumer uses count deltas to know how many NEW rows arrived, then simply takes the TOP new_rows.

Requires:
  pip install numpy pyqt5 pyqtgraph
  + your libemg environment
  + your streamer launcher function (emagerv3_streamer) that starts the process and returns (streamer, smi)
"""

import time
import argparse
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QFont
import pyqtgraph as pg

from libemg.data_handler import OnlineDataHandler
from libemg.streamers import emagerv3_streamer  # must start your streamer process


# -----------------------------
# Shared memory modality names
# -----------------------------
MOD_EMG = "emg"
MOD_EMG_COUNT = "emg_count"

MOD_IMU = "imu"
MOD_IMU_COUNT = "imu_count"

MOD_SAMPLE_ID = "sample_id"
MOD_SAMPLE_ID_COUNT = "sample_id_count"


# -----------------------------
# Rolling buffer for plotting (ring, local only)
# -----------------------------
class RollingBuffer:
    def __init__(self, channels=64, width=4096, dtype=np.float32):
        self.channels = int(channels)
        self.width = int(width)
        self.buf = np.zeros((self.channels, self.width), dtype=dtype)
        self.write_idx = 0
        self.lock = QtCore.QMutex()
        self.valid = False

    def append_block(self, block_ch_time: np.ndarray):
        """Append a (64 x N) block: channels x time."""
        if block_ch_time is None:
            return
        if block_ch_time.ndim != 2 or block_ch_time.shape[0] != self.channels:
            return
        cols = int(block_ch_time.shape[1])
        if cols <= 0:
            return

        self.lock.lock()
        try:
            first = min(cols, self.width - self.write_idx)
            second = cols - first
            self.buf[:, self.write_idx:self.write_idx + first] = block_ch_time[:, :first]
            if second > 0:
                self.buf[:, :second] = block_ch_time[:, first:first + second]
            self.write_idx = (self.write_idx + cols) % self.width
            self.valid = True
        finally:
            self.lock.unlock()

    def chronological_view_copy(self):
        """Return a COPY (channels x width) in chronological order (thread-safe)."""
        self.lock.lock()
        try:
            if not self.valid:
                return None
            if self.write_idx == 0:
                return self.buf.copy()
            return np.concatenate((self.buf[:, self.write_idx:], self.buf[:, :self.write_idx]), axis=1).copy()
        finally:
            self.lock.unlock()


# -----------------------------
# Scale selection dialog
# -----------------------------
class ScaleSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Y Scale")
        self.y_min = 0
        self.y_max = 4092

        title_font = QFont()
        title_font.setPointSize(16)
        button_font = QFont()
        button_font.setPointSize(14)

        label = QtWidgets.QLabel("Choose the amplitude scale for all 64 channels:")
        label.setWordWrap(True)
        label.setFont(title_font)
        label.setAlignment(QtCore.Qt.AlignCenter)

        btn_micro = QtWidgets.QPushButton("Micro (1900–2060)")
        btn_small = QtWidgets.QPushButton("Small (1700–2300)")
        btn_medium = QtWidgets.QPushButton("Medium (1400–2600)")
        btn_large = QtWidgets.QPushButton("Large (1000–3000)")
        btn_full = QtWidgets.QPushButton("Full (0–4092)")

        for b in (btn_micro, btn_small, btn_medium, btn_large, btn_full):
            b.setFont(button_font)
            b.setMinimumHeight(40)

        btn_micro.clicked.connect(lambda: self._choose(1900, 2060))
        btn_small.clicked.connect(lambda: self._choose(1700, 2300))
        btn_medium.clicked.connect(lambda: self._choose(1400, 2600))
        btn_large.clicked.connect(lambda: self._choose(1000, 3000))
        btn_full.clicked.connect(lambda: self._choose(0, 4092))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addSpacing(15)
        layout.addWidget(btn_micro)
        layout.addWidget(btn_small)
        layout.addWidget(btn_medium)
        layout.addWidget(btn_large)
        layout.addWidget(btn_full)

        self.setLayout(layout)
        self.resize(600, 400)

    def _choose(self, y0, y1):
        self.y_min, self.y_max = y0, y1
        self.accept()


# -----------------------------
# GUI grid
# -----------------------------
class LiveGrid64(QtWidgets.QMainWindow):
    def __init__(self, width, fps=30.0, title="Live 64-channel scope (16×4)",
                 y_min=0.0, y_max=4092.0):
        super().__init__()
        self.fps = float(fps)
        self.setWindowTitle(title)
        self.resize(1600, 900)

        self.y_min = float(y_min)
        self.y_max = float(y_max)

        pg.setConfigOptions(
            antialias=False,
            useOpenGL=True,
            enableExperimental=True
        )

        cw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(cw)

        self.plots = []
        self.curves = []
        self.x = np.arange(width, dtype=float)

        rows, cols = 16, 4
        for c in range(cols):
            for r in range(rows):
                ch = r * cols + c
                p = cw.addPlot(row=r, col=c)
                p.showGrid(x=False, y=False, alpha=0.0)
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.hideButtons()
                p.setDownsampling(mode='peak', auto=True)
                p.setClipToView(True)
                p.setLabel('left', f'Ch {ch}', **{'font-size': '8pt'})
                p.setLabel('bottom', '', **{'font-size': '8pt'})
                p.getAxis('left').setStyle(showValues=False)
                p.getAxis('left').setTicks([])
                p.getAxis('bottom').setStyle(showValues=False)
                p.getAxis('bottom').setTicks([])
                p.setXRange(0, width - 1, padding=0)
                p.setYRange(self.y_min, self.y_max)
                vb = p.getViewBox()
                vb.setDefaultPadding(0.0)

                curve = p.plot(pen=pg.mkPen('r', width=1))
                self.plots.append(p)
                self.curves.append(curve)

    @QtCore.pyqtSlot(object)
    def update_from_block(self, view):
        """Slot to receive (channels x width) NumPy array and draw it."""
        if view is None:
            return
        for ch in range(64):
            self.curves[ch].setData(self.x, view[ch], _callSync='off')


# -----------------------------
# Consumer thread (reads from OnlineDataHandler, fills RollingBuffer)
# -----------------------------
class SharedMemoryConsumerVStack(QtCore.QThread):
    """
    Pulls from OnlineDataHandler shared memory and appends new EMG samples to RollingBuffer.

    This is adapted to your NEW streamer behavior:
      - writer prepends new rows to the TOP via vstack((new[::-1], buffer))
      - so the newest data is at buffer[0:new_rows] (in reverse order), and older scrolls down
    We use count deltas to know exactly how many new rows arrived (no ring math).
    """
    status = QtCore.pyqtSignal(str)

    def __init__(self, odh: OnlineDataHandler, rbuf: RollingBuffer, poll_hz=200.0, parent=None):
        super().__init__(parent)
        self.odh = odh
        self.rbuf = rbuf
        self.poll_period = 1.0 / max(1.0, float(poll_hz))
        self._stop = False

        self.last_emg_count = 0
        self.last_imu_count = 0
        self.last_sid_count = 0

        self.last_sample_id = None

    @staticmethod
    def _take_new_from_vstack_buffer(buf: np.ndarray, new_rows: int) -> np.ndarray:
        """
        In your vstack-prepend buffer, new rows are stored at the top:
          buf[0:new_rows] are the newly written rows BUT in reverse order because new[::-1] was used.

        We return them in chronological order (oldest->newest) for local append:
          - newest is at buf[0]
          - oldest of the new chunk is at buf[new_rows-1]
          => reverse again: buf[new_rows-1::-1]
        """
        if new_rows <= 0:
            return np.zeros((0, buf.shape[1]), dtype=buf.dtype)
        new_rows = min(int(new_rows), int(buf.shape[0]))
        chunk_rev = buf[:new_rows, :]              # newest->oldest (within the new chunk)
        chunk = chunk_rev[::-1, :].copy()          # oldest->newest
        return chunk

    def run(self):
        t_next = time.monotonic()
        while not self._stop:
            now = time.monotonic()
            if now < t_next:
                time.sleep(min(0.002, t_next - now))
                continue

            try:
                vals, count = self.odh.get_data()
            except Exception:
                time.sleep(0.01)
                t_next = time.monotonic() + self.poll_period
                continue

            # --- EMG ---
            if MOD_EMG in vals and MOD_EMG in count:
                total_emg = int(count[MOD_EMG][0][0])  # absolute rows ever produced
                new_emg = total_emg - self.last_emg_count
                if new_emg > 0:
                    emg_buf = vals[MOD_EMG]  # (H,64) where TOP contains newest rows
                    chunk = self._take_new_from_vstack_buffer(emg_buf, new_emg)  # (N,64)

                    # Convert to channels x time for plotting
                    block_ch_time = chunk.T.astype(np.float32, copy=False)  # (64,N)
                    self.rbuf.append_block(block_ch_time)

                    self.last_emg_count += int(chunk.shape[0])

            # --- sample_id (optional for status/debug) ---
            if MOD_SAMPLE_ID in vals and MOD_SAMPLE_ID in count:
                total_sid = int(count[MOD_SAMPLE_ID][0][0])
                new_sid = total_sid - self.last_sid_count
                if new_sid > 0:
                    sid_buf = vals[MOD_SAMPLE_ID]  # (H,1)
                    sid_chunk = self._take_new_from_vstack_buffer(sid_buf, new_sid)  # (N,1)
                    if sid_chunk.shape[0] > 0:
                        self.last_sample_id = int(sid_chunk[-1, 0])  # last (newest chronologically)
                    self.last_sid_count += int(sid_chunk.shape[0])

            # --- IMU (optional status display) ---
            if MOD_IMU in vals and MOD_IMU in count:
                total_imu = int(count[MOD_IMU][0][0])
                new_imu = total_imu - self.last_imu_count
                if new_imu > 0:
                    imu_buf = vals[MOD_IMU]  # (H,6)
                    imu_chunk = self._take_new_from_vstack_buffer(imu_buf, new_imu)  # (N,6)
                    if imu_chunk.shape[0] > 0:
                        last_imu = imu_chunk[-1]
                        msg = f"sample_id={self.last_sample_id} | imu={last_imu.tolist()}"
                        self.status.emit(msg)
                    self.last_imu_count += int(imu_chunk.shape[0])

            t_next = time.monotonic() + self.poll_period

    def stop(self):
        self._stop = True


# -----------------------------
# Plot update worker (pushes latest view to GUI at fps)
# -----------------------------
class UpdateWorker(QtCore.QThread):
    newBlock = QtCore.pyqtSignal(object)

    def __init__(self, rbuf: RollingBuffer, fps=30.0, parent=None):
        super().__init__(parent)
        self.rbuf = rbuf
        self.period = 1.0 / max(1.0, float(fps))
        self._stop = False

    def run(self):
        next_t = time.monotonic()
        while not self._stop:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            view = self.rbuf.chronological_view_copy()
            self.newBlock.emit(view)
            next_t = time.monotonic() + self.period

    def stop(self):
        self._stop = True


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="PyQtGraph 64-ch live view using LibEMG shared memory (vstack-prepend buffer streamer).")
    ap.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0=run forever)")
    ap.add_argument("--plot-window", type=int, default=6144, help="Samples per channel kept on screen")
    ap.add_argument("--plot-fps", type=float, default=20.0, help="Plot update rate (Hz)")
    ap.add_argument("--poll-hz", type=float, default=300.0, help="Shared-memory poll rate (Hz)")

    # streamer args (optional)
    ap.add_argument("--baud", type=int, default=3000000, help="Baud rate (passed to emagerv3_streamer)")

    args = ap.parse_args()

    # 1) Start streamer (separate process)
    streamer, smi = emagerv3_streamer(baud_rate=args.baud)
    print("[INFO] Streamer started.")
    print(f"[INFO] shared_memory_items tags: {[x[0] for x in smi]}")

    # 2) Create OnlineDataHandler
    odh = OnlineDataHandler(shared_memory_items=smi)

    # 3) App + scale dialog
    app = QtWidgets.QApplication([])

    scale_dialog = ScaleSelectionDialog()
    result = scale_dialog.exec_()
    if result != QtWidgets.QDialog.Accepted:
        y_min, y_max = 0.0, 4092.0
    else:
        y_min, y_max = float(scale_dialog.y_min), float(scale_dialog.y_max)
    print(f"[INFO] Using Y-range: {y_min} to {y_max}")

    # 4) Rolling buffer, window, workers
    rbuf = RollingBuffer(channels=64, width=args.plot_window, dtype=np.float32)

    win = LiveGrid64(width=args.plot_window,
                     fps=args.plot_fps,
                     title="Live 64-channel scope (16×4) [EMaGer v3 streamer | vstack buffer]",
                     y_min=y_min,
                     y_max=y_max)
    win.show()

    updater = UpdateWorker(rbuf=rbuf, fps=args.plot_fps)
    updater.newBlock.connect(win.update_from_block, QtCore.Qt.QueuedConnection)
    updater.start()

    consumer = SharedMemoryConsumerVStack(odh=odh, rbuf=rbuf, poll_hz=args.poll_hz)
    consumer.status.connect(lambda s: win.setWindowTitle(f"Live 64-ch [EMaGer v3 vstack] | {s}"))
    consumer.start()

    # optional duration stop
    stop_timer = None
    if args.duration and args.duration > 0:
        def stop_after():
            win.close()
        stop_timer = QtCore.QTimer()
        stop_timer.setSingleShot(True)
        stop_timer.timeout.connect(stop_after)
        stop_timer.start(int(args.duration * 1000))

    # 5) Run + cleanup
    ret = 0
    try:
        ret = app.exec_()
    finally:
        try:
            consumer.stop()
            consumer.wait(1000)
        except Exception:
            pass
        try:
            updater.stop()
            updater.wait(1000)
        except Exception:
            pass

        # stop streamer process
        try:
            streamer.stop()  # your streamer has stop()
        except Exception:
            try:
                streamer.terminate()
            except Exception:
                pass

    return ret


if __name__ == "__main__":
    main()
