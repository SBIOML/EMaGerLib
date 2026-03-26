#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 6-axis IMU plotting using PyQtGraph, with data coming from LibEMG shared memory
via your EmagerV3Streamer (buffer_write via vstack prepend).

This version:
- shows proper units on screen
- uses dynamic Y autoscaling so each plot fills the available space better
- keeps accelerometer and gyroscope displays clearer and easier to read

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
import pyqtgraph as pg

from libemg.data_handler import OnlineDataHandler
from libemg.streamers import emagerv3_streamer


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
# Rolling buffer for plotting
# -----------------------------
class RollingBuffer:
    def __init__(self, channels=6, width=4096, dtype=np.float32):
        self.channels = int(channels)
        self.width = int(width)
        self.buf = np.zeros((self.channels, self.width), dtype=dtype)
        self.write_idx = 0
        self.lock = QtCore.QMutex()
        self.valid = False

    def append_block(self, block_ch_time: np.ndarray):
        """Append a (channels x N) block. If N > width, keep only the newest width samples."""
        if block_ch_time is None:
            return
        if block_ch_time.ndim != 2 or block_ch_time.shape[0] != self.channels:
            return

        cols = int(block_ch_time.shape[1])
        if cols <= 0:
            return

        if cols > self.width:
            block_ch_time = block_ch_time[:, -self.width:]
            cols = self.width

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
        """Return COPY of buffer in chronological order: shape (channels x width)."""
        self.lock.lock()
        try:
            if not self.valid:
                return None
            if self.write_idx == 0:
                return self.buf.copy()
            return np.concatenate(
                (self.buf[:, self.write_idx:], self.buf[:, :self.write_idx]),
                axis=1
            ).copy()
        finally:
            self.lock.unlock()


# -----------------------------
# GUI for 6 IMU axes
# -----------------------------
class LiveIMU6(QtWidgets.QMainWindow):
    def __init__(self, width, fps=30.0, title="Live 6-axis IMU scope",
                 accel_unit="mg", gyro_unit="raw", auto_scale=True):
        super().__init__()
        self.fps = float(fps)
        self.setWindowTitle(title)
        self.resize(1500, 950)

        self.accel_unit = accel_unit
        self.gyro_unit = gyro_unit
        self.auto_scale = auto_scale

        # Minimum half-ranges to avoid over-zooming when signal is nearly flat
        self.min_half_ranges = [300, 300, 300, 200, 200, 200]

        # Initial Y ranges
        self.current_y_ranges = [
            [-1000, 1000], [-1000, 1000], [-1000, 1000],
            [-500, 500], [-500, 500], [-500, 500]
        ]

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

        # ch, row, col, title, color, unit
        plot_defs = [
            (0, 0, 0, "Accelerometer X", 'r', self.accel_unit),
            (1, 1, 0, "Accelerometer Y", 'g', self.accel_unit),
            (2, 2, 0, "Accelerometer Z", 'b', self.accel_unit),
            (3, 0, 1, "Gyroscope X", 'r', self.gyro_unit),
            (4, 1, 1, "Gyroscope Y", 'g', self.gyro_unit),
            (5, 2, 1, "Gyroscope Z", 'b', self.gyro_unit),
        ]

        for ch, r, c, title_txt, color, unit in plot_defs:
            p = cw.addPlot(row=r, col=c)
            p.setTitle(title_txt)
            p.showGrid(x=False, y=True, alpha=0.2)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.setDownsampling(mode='peak', auto=True)
            p.setClipToView(True)
            p.setLabel('left', unit, **{'font-size': '11pt'})
            p.setLabel('bottom', '', **{'font-size': '8pt'})
            p.getAxis('bottom').setStyle(showValues=False)
            p.getAxis('bottom').setTicks([])
            p.setXRange(0, width - 1, padding=0)

            vb = p.getViewBox()
            vb.setDefaultPadding(0.0)

            curve = p.plot(pen=pg.mkPen(color, width=2))
            self.plots.append(p)
            self.curves.append(curve)

    def _update_y_range(self, ch, y):
        """Auto-scale one channel with smoothing and margin."""
        y = np.asarray(y)
        if y.size == 0:
            return

        finite = np.isfinite(y)
        if not np.any(finite):
            return

        yv = y[finite]
        ymin = float(np.min(yv))
        ymax = float(np.max(yv))

        center = 0.5 * (ymin + ymax)
        half = 0.5 * (ymax - ymin)

        # Add 15% margin
        half *= 1.15

        # Prevent absurd zoom when nearly flat
        half = max(half, self.min_half_ranges[ch])

        target_low = center - half
        target_high = center + half

        old_low, old_high = self.current_y_ranges[ch]

        # Smoothing so scale does not jump too aggressively
        alpha = 0.18
        new_low = (1 - alpha) * old_low + alpha * target_low
        new_high = (1 - alpha) * old_high + alpha * target_high

        self.current_y_ranges[ch] = [new_low, new_high]
        self.plots[ch].setYRange(new_low, new_high, padding=0)

    @QtCore.pyqtSlot(object)
    def update_from_block(self, view):
        """Receive (6 x width) NumPy array and draw it."""
        if view is None:
            return

        for ch in range(6):
            y = view[ch]
            self.curves[ch].setData(self.x, y, _callSync='off')
            if self.auto_scale:
                self._update_y_range(ch, y)


# -----------------------------
# Consumer thread for IMU
# -----------------------------
class SharedMemoryConsumerVStackIMU(QtCore.QThread):
    """
    Pulls IMU data from OnlineDataHandler shared memory and appends new samples to RollingBuffer.

    Adapted to vstack-prepend writer behavior:
      - writer prepends new rows to the TOP via vstack((new[::-1], buffer))
      - newest data is therefore at buffer[0:new_rows] in reverse order
    """
    status = QtCore.pyqtSignal(str)

    def __init__(self, odh: OnlineDataHandler, rbuf: RollingBuffer, poll_hz=200.0,
                 accel_to_g=False, parent=None):
        super().__init__(parent)
        self.odh = odh
        self.rbuf = rbuf
        self.poll_period = 1.0 / max(1.0, float(poll_hz))
        self._stop = False

        self.last_imu_count = 0
        self.last_sample_id_count = 0
        self.last_sample_id = None

        self.accel_to_g = accel_to_g

    @staticmethod
    def _take_new_from_vstack_buffer(buf: np.ndarray, new_rows: int) -> np.ndarray:
        """
        In the vstack-prepend buffer:
          buf[0:new_rows] are the newly written rows, but in reverse order.

        Return them in chronological order (oldest -> newest).
        """
        if new_rows <= 0:
            return np.zeros((0, buf.shape[1]), dtype=buf.dtype)

        new_rows = min(int(new_rows), int(buf.shape[0]))
        chunk_rev = buf[:new_rows, :]      # newest -> oldest
        chunk = chunk_rev[::-1, :].copy()  # oldest -> newest
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

            # --- IMU ---
            if MOD_IMU in vals and MOD_IMU in count:
                total_imu = int(count[MOD_IMU][0][0])
                new_imu = total_imu - self.last_imu_count

                if new_imu > 0:
                    imu_buf = vals[MOD_IMU]  # expected shape: (H, 6)
                    chunk = self._take_new_from_vstack_buffer(imu_buf, new_imu)  # (N,6)

                    # Swap the two bytes of every int16 IMU value
                    chunk = np.asarray(chunk, dtype=np.int16).byteswap()

                    block_ch_time = chunk.T.astype(np.float32, copy=False)  # (6,N)

                    # Optional: convert accelerometer from mg to g
                    if self.accel_to_g:
                        block_ch_time[0:3, :] /= 1000.0

                    self.rbuf.append_block(block_ch_time)
                    self.last_imu_count += int(chunk.shape[0])

                    if chunk.shape[0] > 0:
                        last_imu = block_ch_time[:, -1]
                        msg = (
                            f"sample_id={self.last_sample_id} | "
                            f"acc=[{last_imu[0]:.2f}, {last_imu[1]:.2f}, {last_imu[2]:.2f}] | "
                            f"gyro=[{last_imu[3]:.2f}, {last_imu[4]:.2f}, {last_imu[5]:.2f}]"
                        )
                        self.status.emit(msg)

            # --- sample_id (optional) ---
            if MOD_SAMPLE_ID in vals and MOD_SAMPLE_ID in count:
                total_sid = int(count[MOD_SAMPLE_ID][0][0])
                new_sid = total_sid - self.last_sample_id_count

                if new_sid > 0:
                    sid_buf = vals[MOD_SAMPLE_ID]  # (H,1)
                    sid_chunk = self._take_new_from_vstack_buffer(sid_buf, new_sid)

                    if sid_chunk.shape[0] > 0:
                        self.last_sample_id = int(sid_chunk[-1, 0])

                    self.last_sample_id_count += int(sid_chunk.shape[0])

            t_next = time.monotonic() + self.poll_period

    def stop(self):
        self._stop = True


# -----------------------------
# Plot update worker
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
    ap = argparse.ArgumentParser(
        description="PyQtGraph live 6-axis IMU viewer using LibEMG shared memory (vstack-prepend buffer streamer)."
    )
    ap.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = run forever)")
    ap.add_argument("--plot-window", type=int, default=200, help="Samples per axis kept on screen")
    ap.add_argument("--plot-fps", type=float, default=20.0, help="Plot update rate (Hz)")
    ap.add_argument("--poll-hz", type=float, default=300.0, help="Shared-memory poll rate (Hz)")
    ap.add_argument("--baud", type=int, default=3000000, help="Baud rate passed to emagerv3_streamer")

    # Units / scaling options
    ap.add_argument("--accel-unit", type=str, default="mg", choices=["mg", "g"],
                    help="Display accelerometer as mg or g")
    ap.add_argument("--gyro-unit", type=str, default="raw",
                    help="Gyroscope unit label to display (e.g. raw, dps, mdps)")

    args = ap.parse_args()

    accel_to_g = (args.accel_unit == "g")

    # 1) Start streamer
    streamer, smi = emagerv3_streamer(baud_rate=args.baud)
    print("[INFO] Streamer started.")
    print(f"[INFO] shared_memory_items tags: {[x[0] for x in smi]}")

    # 2) Create OnlineDataHandler
    odh = OnlineDataHandler(shared_memory_items=smi)

    # 3) Qt app
    app = QtWidgets.QApplication([])

    # 4) Rolling buffer, window, workers
    rbuf = RollingBuffer(channels=6, width=args.plot_window, dtype=np.float32)

    win = LiveIMU6(
        width=args.plot_window,
        fps=args.plot_fps,
        title="Live 6-axis IMU scope [EMaGer v3 streamer | vstack buffer]",
        accel_unit=args.accel_unit,
        gyro_unit=args.gyro_unit,
        auto_scale=True
    )
    win.show()

    updater = UpdateWorker(rbuf=rbuf, fps=args.plot_fps)
    updater.newBlock.connect(win.update_from_block, QtCore.Qt.QueuedConnection)
    updater.start()

    consumer = SharedMemoryConsumerVStackIMU(
        odh=odh,
        rbuf=rbuf,
        poll_hz=args.poll_hz,
        accel_to_g=accel_to_g
    )
    consumer.status.connect(lambda s: win.setWindowTitle(f"Live 6-axis IMU | {s}"))
    consumer.start()

    # Optional duration stop
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

        try:
            streamer.stop()
        except Exception:
            try:
                streamer.terminate()
            except Exception:
                pass

    return ret


if __name__ == "__main__":
    main()