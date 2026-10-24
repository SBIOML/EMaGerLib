#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 32-channel plotting (8x4) using PyQtGraph, but data comes from LibEMG shared memory
via your EmagerPuckStreamer (buffer_write via vstack prepend) rather than reading serial here.

IMPORTANT:
- Your streamer writes NEW data by PREPENDING (vstack) into the shared buffer.
- Therefore the shared memory array is NOT a classical ring. The newest rows are at the TOP.
- Counts (tag_count) still increase by number of rows written.
- This consumer uses count deltas to know how many NEW rows arrived, then simply takes the TOP new_rows.

Requires:
  pip install numpy pyqt5 pyqtgraph
  + your libemg environment
  + your streamer launcher function (emagerPuck_streamer) that starts the process and returns (streamer, smi)
"""

import time
import argparse
import numpy as np

# Import libemg (and its transitive torch dep, since libemg PR #130) BEFORE any
# Qt binding. On Windows, loading torch after a Qt binding makes c10.dll's DLL
# loader pick up Qt's shipped runtimes from PATH, producing WinError 1114.
from libemg.data_handler import OnlineDataHandler
from libemg.streamers import emagerPuck_streamer  # must start your streamer process

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QFont
import pyqtgraph as pg


# -----------------------------
# Shared memory modality names
# -----------------------------
MOD_EMG = "emg"
MOD_EMG_COUNT = "emg_count"



# -----------------------------
# Rolling buffer for plotting (ring, local only)
# -----------------------------
class RollingBuffer:
    def __init__(self, channels=32, width=4096, dtype=np.float32):
        self.channels = int(channels)
        self.width = int(width)
        self.buf = np.zeros((self.channels, self.width), dtype=dtype)
        self.write_idx = 0
        self.lock = QtCore.QMutex()
        self.valid = False

    def append_block(self, block_ch_time: np.ndarray):
        """Append a (32 x N) block: channels x time."""
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

        label = QtWidgets.QLabel("Choose the amplitude scale for all 32 channels:")
        label.setWordWrap(True)
        label.setFont(title_font)
        label.setAlignment(QtCore.Qt.AlignCenter)

        btn_micro = QtWidgets.QPushButton("Micro (4000)")
        btn_small = QtWidgets.QPushButton("Small (7000)")
        btn_medium = QtWidgets.QPushButton("Medium (10000)")
        btn_large = QtWidgets.QPushButton("Large (20000)")
        btn_full = QtWidgets.QPushButton("Full (32767)")

        for b in (btn_micro, btn_small, btn_medium, btn_large, btn_full):
            b.setFont(button_font)
            b.setMinimumHeight(40)

        btn_micro.clicked.connect(lambda: self._choose(-4000, 4000))
        btn_small.clicked.connect(lambda: self._choose(-7000, 7000))
        btn_medium.clicked.connect(lambda: self._choose(-10000, 10000))
        btn_large.clicked.connect(lambda: self._choose(-20000, 20000))
        btn_full.clicked.connect(lambda: self._choose(-32768, 32767))

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
class LiveGrid32(QtWidgets.QMainWindow):
    def __init__(self, width, rbuf: RollingBuffer, fps=40.0, title="Live scope (8×4)",
                 y_min=0.0, y_max=4092.0, n_channels=32, plot_decimate=2):
        super().__init__()
        self.fps = float(fps)
        self.rbuf = rbuf
        self.n_channels = int(n_channels)
        self.plot_decimate = max(1, int(plot_decimate))

        self.setWindowTitle(title)
        self.resize(1600, 900)

        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # These options help PyQtGraph draw faster.
        # Note: useOpenGL helps on some machines but not all. If the graph looks strange,
        # set useOpenGL=False and try again.
        pg.setConfigOptions(
            antialias=False,
            useOpenGL=True,
            enableExperimental=True
        )

        cw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(cw)

        self.plots = []
        self.curves = []

        # Plot fewer points than the acquisition buffer contains.
        # This keeps the visual shape while reducing GUI load.
        self.x = np.arange(0, width, self.plot_decimate, dtype=np.float32)

        rows, cols = 8, 4

        for ch in range(self.n_channels):
            r = ch // cols
            c = ch % cols

            p = cw.addPlot(row=r, col=c)
            p.showGrid(x=False, y=False, alpha=0.0)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()

            # Important for speed when many curves are displayed.
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode='peak')

            p.setLabel('left', f'Ch {ch}', **{'font-size': '8pt'})
            p.setLabel('bottom', '', **{'font-size': '8pt'})
            p.getAxis('left').setStyle(showValues=False)
            p.getAxis('left').setTicks([])
            p.getAxis('bottom').setStyle(showValues=False)
            p.getAxis('bottom').setTicks([])
            p.setXRange(0, width - 1, padding=0)
            p.setYRange(self.y_min, self.y_max)
            p.getViewBox().setDefaultPadding(0.0)

            curve = p.plot(pen=pg.mkPen('r', width=1))
            self.plots.append(p)
            self.curves.append(curve)

        # Use a GUI-thread timer instead of a QThread emitting many queued updates.
        # This avoids update backlog, which is a common cause of laggy plots.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(max(1, int(1000.0 / max(1.0, self.fps))))

    @QtCore.pyqtSlot()
    def refresh_plot(self):
        view = self.rbuf.chronological_view_copy()
        self.update_from_block(view)

    def update_from_block(self, view):
        """Receive channels x width NumPy array and draw it."""
        if view is None:
            return

        n_ch = min(view.shape[0], len(self.curves))
        for ch in range(n_ch):
            y = view[ch, ::self.plot_decimate]
            self.curves[ch].setData(self.x[:len(y)], y, _callSync='off')


# -----------------------------
# Consumer thread (reads from OnlineDataHandler, fills RollingBuffer)
# -----------------------------
class SharedMemoryConsumerVStack(QtCore.QThread):
    """
    Pulls from OnlineDataHandler shared memory and appends new EMG samples to RollingBuffer.

    This is adapted to the streamer behavior:
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
        initialized = False

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

            # First read: sync to "now" so we do not ingest old backlog
            if not initialized:
                if MOD_EMG in count:
                    self.last_emg_count = int(count[MOD_EMG][0][0])

                initialized = True
                t_next = time.monotonic() + self.poll_period
                continue

            # --- EMG handling ---
            if MOD_EMG in vals and MOD_EMG in count:
                total_emg = int(count[MOD_EMG][0][0])
                new_emg = total_emg - self.last_emg_count

                if new_emg > 0:
                    emg_buf = vals[MOD_EMG]  # expected shape: (rows, 32)
                    new_chunk = self._take_new_from_vstack_buffer(emg_buf, new_emg)

                    if new_chunk is not None and new_chunk.size > 0:
                        # convert (time, ch) -> (ch, time) for plotting
                        block_ch_time = new_chunk.T
                        self.rbuf.append_block(block_ch_time)

                    self.last_emg_count = total_emg

            t_next = time.monotonic() + self.poll_period
    def stop(self):
        self._stop = True


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="PyQtGraph 32-ch live view using LibEMG shared memory (vstack-prepend buffer streamer).")
    ap.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0=run forever)")
    ap.add_argument("--plot-window", type=int, default=2048, help="Samples per channel kept on screen")
    ap.add_argument("--plot-fps", type=float, default=40.0, help="GUI redraw rate (Hz)")
    ap.add_argument("--plot-decimate", type=int, default=2, help="Plot every Nth sample to reduce lag")
    ap.add_argument("--poll-hz", type=float, default=300.0, help="Shared-memory poll rate (Hz)")

    # streamer args (optional)
    ap.add_argument("--baud", type=int, default=3000000, help="Baud rate (passed to emagerPuck_streamer)")

    args = ap.parse_args()

    # 1) Start streamer (separate process)
    streamer, smi = emagerPuck_streamer(baud_rate=args.baud, debug=True)
    print("[INFO] Streamer started.")
    print(f"[INFO] shared_memory_items tags: {[x[0] for x in smi]}")

    # 2) Create OnlineDataHandler
    odh = OnlineDataHandler(shared_memory_items=smi)

    # 3) App + scale dialog
    app = QtWidgets.QApplication([])

    scale_dialog = ScaleSelectionDialog()
    result = scale_dialog.exec_()
    if result != QtWidgets.QDialog.Accepted:
        print("[INFO] Scale selection cancelled, using Full (-32768–32767).")
        y_min, y_max = -32768.0, 32767.0
    else:
        y_min, y_max = float(scale_dialog.y_min), float(scale_dialog.y_max)
    print(f"[INFO] Using Y-range: {y_min} to {y_max}")

    # 4) Rolling buffer, window, workers
    rbuf = RollingBuffer(channels=32, width=args.plot_window, dtype=np.float32)

    win = LiveGrid32(width=args.plot_window,
                     rbuf=rbuf,
                     fps=args.plot_fps,
                     title="Live 32-channel scope (8×4) [EMaGer Puck streamer | vstack buffer]",
                     y_min=y_min,
                     y_max=y_max,
                     n_channels=32,
                     plot_decimate=args.plot_decimate)
    win.show()

    consumer = SharedMemoryConsumerVStack(odh=odh, rbuf=rbuf, poll_hz=args.poll_hz)
    consumer.status.connect(lambda s: win.setWindowTitle(f"Live 32-ch [EMaGer Puck vstack] | {s}"))
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
