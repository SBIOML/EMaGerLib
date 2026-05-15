#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 64-channel EMG heatmap using LibEMG shared memory + emagerv3_streamer.

This script combines:
1) the shared-memory EMG acquisition approach from the live 64-channel viewer
2) the heatmap / intensity / calibration logic from the intensity-map script

Key points:
- Data comes from emagerv3_streamer -> shared memory -> OnlineDataHandler
- Streamer writes new rows by PREPENDING to the shared buffer (vstack behavior)
- We use count deltas to recover exactly the newly written rows
- A local rolling buffer stores recent samples for all 64 channels
- Intensity per channel is computed as mean(abs(x - mean(x))) over the last N samples
- Heatmap can be calibrated with:
    Phase 1: max flexion
    Phase 2: max extension
    Phase 3: relaxed / noise

Requirements:
    pip install numpy pyqt5 pyqtgraph
    + your libemg environment
    + emagerv3_streamer available in libemg.streamers
"""

import time
import argparse
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QFont
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
# Display scaling tweak
# -----------------------------
DISPLAY_SCALE_FACTOR = 0.85


# -----------------------------
# Rolling buffer (channels x time)
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
        """
        Append a (channels x N) block.
        """
        if block_ch_time is None:
            return
        if block_ch_time.ndim != 2 or block_ch_time.shape[0] != self.channels:
            return

        cols = int(block_ch_time.shape[1])
        if cols <= 0:
            return

        # If incoming block is bigger than buffer, only keep the newest samples
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
        """
        Return a copy of the rolling buffer in chronological order: (channels x width)
        """
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
# Colormap helpers
# -----------------------------
def build_lut_from_pg(name="viridis", n=256):
    try:
        cmap = pg.colormap.get(name)
        lut = (cmap.getLookupTable(0.0, 1.0, n)[:, :3]).astype(np.ubyte)
        return lut
    except Exception:
        return None


def build_lut_fallback_turbo(n=256):
    stops = [
        (0.00, (0, 7, 100)),
        (0.13, (0, 131, 184)),
        (0.25, (0, 199, 140)),
        (0.37, (77, 220, 78)),
        (0.50, (194, 223, 35)),
        (0.62, (255, 190, 0)),
        (0.75, (255, 113, 0)),
        (0.87, (231, 0, 0)),
        (1.00, (125, 0, 0)),
    ]
    pos = np.array([p for p, _ in stops], dtype=float)
    cols = np.array([c for _, c in stops], dtype=float)
    x = np.linspace(0, 1, n)
    lut = np.empty((n, 3), dtype=float)
    for ch in range(3):
        lut[:, ch] = np.interp(x, pos, cols[:, ch])
    return lut.clip(0, 255).astype(np.ubyte)


def get_colormap_lut(name="viridis", n=256):
    lut = build_lut_from_pg(name, n)
    if lut is not None:
        return lut
    return build_lut_fallback_turbo(n)


# -----------------------------
# Intensity helper
# -----------------------------
def compute_intensity_snapshot(view, N):
    """
    view: (channels x width)
    intensity = mean(abs(x - mean(x))) over last N samples for each channel
    """
    if view is None:
        return None

    W = view.shape[1]
    n = min(int(N), W)
    if n <= 0:
        return None

    last = view[:, -n:]
    mean = last.mean(axis=1, keepdims=True)
    centered = last - mean
    intens = np.mean(np.abs(centered), axis=1)
    return intens


# -----------------------------
# Calibration dialog
# -----------------------------
class CalibrationDialog(QtWidgets.QDialog):
    """
    3-phase calibration:
      1) max flexion
      2) max extension
      3) relaxed / noise
    """
    def __init__(
        self,
        rbuf: RollingBuffer,
        intensity_window: int,
        flex_duration: float = 1.5,
        ext_duration: float = 1.5,
        noise_duration: float = 1.5,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.rbuf = rbuf
        self.intensity_window = int(max(1, intensity_window))

        self.phase_index = 0
        self.phase_names = ["Max FLEXION", "Max EXTENSION", "Relaxed (noise)"]
        self.phase_durations = [flex_duration, ext_duration, noise_duration]

        self.max_flex = 0.0
        self.max_ext = 0.0
        self.noise_floor = 0.0

        self._phase_start_time = None
        self._noise_samples = []

        self.label = QtWidgets.QLabel(self._phase_text())
        self.label.setWordWrap(True)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.start_button = QtWidgets.QPushButton("Start")
        self.cancel_button = QtWidgets.QPushButton("Cancel")

        big_font = QFont()
        big_font.setPointSize(14)
        self.label.setFont(big_font)
        self.progress.setFont(big_font)
        self.start_button.setFont(big_font)
        self.cancel_button.setFont(big_font)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.cancel_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.resize(700, 280)

        self.start_button.clicked.connect(self.start_phase)
        self.cancel_button.clicked.connect(self.reject)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._tick)

    def _phase_text(self):
        name = self.phase_names[self.phase_index]
        duration = self.phase_durations[self.phase_index]
        if self.phase_index == 0:
            extra = "When you press Start, FLEX as hard as you can and hold until the bar reaches 100%."
        elif self.phase_index == 1:
            extra = "When you press Start, EXTEND as hard as you can and hold until the bar reaches 100%."
        else:
            extra = "When you press Start, stay completely RELAXED and still until the bar reaches 100%."
        return (
            f"Phase {self.phase_index + 1}/3: {name}\n\n"
            f"{extra}\n(Duration: {duration:.1f} s)"
        )

    def start_phase(self):
        self.progress.setValue(0)
        self._phase_start_time = time.monotonic()
        if self.phase_index == 2:
            self._noise_samples = []
        self.timer.start()
        self.start_button.setEnabled(False)

    def _tick(self):
        if self._phase_start_time is None:
            return

        now = time.monotonic()
        elapsed = now - self._phase_start_time
        duration = self.phase_durations[self.phase_index]

        pct = max(0.0, min(1.0, elapsed / duration)) * 100.0
        self.progress.setValue(int(pct))

        view = self.rbuf.chronological_view_copy()
        if view is not None:
            intens = compute_intensity_snapshot(view, self.intensity_window)
            if intens is not None:
                if self.phase_index in (0, 1):
                    val = float(np.max(intens))
                    if self.phase_index == 0:
                        if val > self.max_flex:
                            self.max_flex = val
                    else:
                        if val > self.max_ext:
                            self.max_ext = val
                else:
                    self._noise_samples.append(intens)

        if elapsed >= duration:
            self.timer.stop()
            self.start_button.setEnabled(True)
            self._phase_start_time = None

            if self.phase_index == 2:
                if self._noise_samples:
                    all_vals = np.concatenate(self._noise_samples)
                    self.noise_floor = float(np.percentile(all_vals, 99.0))
                else:
                    self.noise_floor = 0.0
                self.accept()
            else:
                self.phase_index += 1
                self.label.setText(self._phase_text())
                self.progress.setValue(0)


# -----------------------------
# Heatmap GUI
# -----------------------------
class LiveIntensityMap(QtWidgets.QMainWindow):
    def __init__(
        self,
        fps=30.0,
        rows=16,
        cols=4,
        cmap_name="viridis",
        title="Live 64-channel EMG Heatmap",
        max_intensity=None
    ):
        super().__init__()
        self.fps = float(fps)
        self.rows = int(rows)
        self.cols = int(cols)
        self.cmap_name = cmap_name
        self.max_intensity = max_intensity
        self.levels_set = False

        self.setWindowTitle(title)
        self.resize(1000, 1200)

        pg.setConfigOptions(
            antialias=False,
            useOpenGL=True,
            enableExperimental=True
        )

        cw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(cw)

        self.plot = cw.addPlot(row=0, col=0)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.getAxis("left").setStyle(showValues=False)
        self.plot.getAxis("left").setTicks([])
        self.plot.getAxis("bottom").setStyle(showValues=False)
        self.plot.getAxis("bottom").setTicks([])
        self.plot.invertY(True)
        self.plot.setAspectLocked(True, ratio=1)

        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)

        lut = get_colormap_lut(self.cmap_name, n=256)
        self.img_item.setLookupTable(lut)

    @QtCore.pyqtSlot(object)
    def update_from_intensity(self, img2d):
        if img2d is None:
            return

        self.img_item.setImage(img2d, autoLevels=False)

        if not self.levels_set:
            if self.max_intensity is not None and self.max_intensity > 0:
                display_max = self.max_intensity * DISPLAY_SCALE_FACTOR
                if display_max <= 0:
                    display_max = self.max_intensity
                self.img_item.setLevels((0.0, display_max))
            else:
                levels = self.img_item.getLevels()
                if levels is not None:
                    self.img_item.setLevels(levels)
            self.levels_set = True


# -----------------------------
# Shared memory consumer
# -----------------------------
class SharedMemoryConsumerVStack(QtCore.QThread):
    """
    Reads from OnlineDataHandler where the streamer writes new rows
    at the TOP of the shared buffer using a vstack-prepend strategy.
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
        Shared memory buffer layout:
        - newest data is at buf[0]
        - top new_rows rows are the newest chunk but reversed within the chunk
        So reverse again to recover chronological order.
        """
        if new_rows <= 0:
            return np.zeros((0, buf.shape[1]), dtype=buf.dtype)

        new_rows = min(int(new_rows), int(buf.shape[0]))
        chunk_rev = buf[:new_rows, :]       # newest -> oldest
        chunk = chunk_rev[::-1, :].copy()   # oldest -> newest
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
                total_emg = int(count[MOD_EMG][0][0])
                new_emg = total_emg - self.last_emg_count

                if new_emg > 0:
                    emg_buf = vals[MOD_EMG]  # (H,64), top contains newest rows
                    chunk = self._take_new_from_vstack_buffer(emg_buf, new_emg)  # (N,64)

                    # Convert to channels x time
                    block_ch_time = chunk.T.astype(np.float32, copy=False)  # (64,N)
                    self.rbuf.append_block(block_ch_time)

                    self.last_emg_count += int(chunk.shape[0])

            # --- sample_id ---
            if MOD_SAMPLE_ID in vals and MOD_SAMPLE_ID in count:
                total_sid = int(count[MOD_SAMPLE_ID][0][0])
                new_sid = total_sid - self.last_sid_count

                if new_sid > 0:
                    sid_buf = vals[MOD_SAMPLE_ID]
                    sid_chunk = self._take_new_from_vstack_buffer(sid_buf, new_sid)
                    if sid_chunk.shape[0] > 0:
                        self.last_sample_id = int(sid_chunk[-1, 0])
                    self.last_sid_count += int(sid_chunk.shape[0])

            # --- IMU ---
            if MOD_IMU in vals and MOD_IMU in count:
                total_imu = int(count[MOD_IMU][0][0])
                new_imu = total_imu - self.last_imu_count

                if new_imu > 0:
                    imu_buf = vals[MOD_IMU]
                    imu_chunk = self._take_new_from_vstack_buffer(imu_buf, new_imu)
                    if imu_chunk.shape[0] > 0:
                        last_imu = imu_chunk[-1]
                        msg = f"sample_id={self.last_sample_id} | imu={last_imu.tolist()}"
                        self.status.emit(msg)
                    self.last_imu_count += int(imu_chunk.shape[0])

            t_next = time.monotonic() + self.poll_period

    def stop(self):
        self._stop = True


# -----------------------------
# Heatmap update worker
# -----------------------------
class HeatmapUpdateWorker(QtCore.QThread):
    newImage = QtCore.pyqtSignal(object)

    def __init__(
        self,
        rbuf: RollingBuffer,
        fps=30.0,
        intensity_window=400,
        rows=16,
        cols=4,
        noise_floor: float = 0.0,
        parent=None
    ):
        super().__init__(parent)
        self.rbuf = rbuf
        self.period = 1.0 / max(1.0, float(fps))
        self.N = int(max(1, intensity_window))
        self.rows = int(rows)
        self.cols = int(cols)
        self.noise_floor = float(noise_floor)
        self._stop = False

    def run(self):
        next_t = time.monotonic()

        while not self._stop:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue

            view = self.rbuf.chronological_view_copy()
            img = None

            if view is not None:
                intens = compute_intensity_snapshot(view, self.N)
                if intens is not None:
                    threshold = self.noise_floor if self.noise_floor > 0 else 5.0
                    intens = intens.copy()
                    intens[intens < threshold] = 0.0

                    num = self.rows * self.cols
                    intens = intens[:num]

                    if intens.shape[0] < num:
                        padded = np.zeros(num, dtype=np.float32)
                        padded[:intens.shape[0]] = intens
                        intens = padded

                    # # Keep channel order simple:
                    # # channel 0..63 reshaped exactly as rows x cols
                    # img = intens.reshape(self.rows, self.cols).astype(np.float32)
                    # Swap X and Y on display
                    img = intens.reshape(self.rows, self.cols).T.astype(np.float32)

            self.newImage.emit(img)
            next_t += self.period

    def stop(self):
        self._stop = True


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Live 64-channel EMG heatmap using LibEMG shared memory + emagerv3_streamer"
    )

    # Runtime
    ap.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0=run forever)")
    ap.add_argument("--baud", type=int, default=3000000, help="Baud rate passed to emagerv3_streamer")

    # Buffer / rates
    ap.add_argument("--plot-window", type=int, default=4096, help="Samples per channel kept in local rolling buffer")
    ap.add_argument("--plot-fps", type=float, default=30.0, help="Heatmap update rate (Hz)")
    ap.add_argument("--poll-hz", type=float, default=300.0, help="Shared-memory poll rate (Hz)")
    ap.add_argument("--intensity-window", type=int, default=400, help="Samples used per channel for intensity")

    # Heatmap layout
    ap.add_argument("--rows", type=int, default=4, help="Heatmap rows")
    ap.add_argument("--cols", type=int, default=16, help="Heatmap cols")
    ap.add_argument("--cmap", type=str, default="viridis", help="Colormap name")

    # Calibration
    ap.add_argument("--no-calibration", action="store_true", help="Skip flex/ext/noise calibration")
    ap.add_argument("--default-noise-floor", type=float, default=5.0, help="Used only if calibration is skipped")
    ap.add_argument("--default-max-intensity", type=float, default=None,
                    help="Optional fixed max intensity if calibration is skipped")

    args = ap.parse_args()

    if args.rows * args.cols < 64:
        print(f"[WARN] rows*cols = {args.rows * args.cols} < 64, so not all channels will be shown.")
    elif args.rows * args.cols > 64:
        print(f"[WARN] rows*cols = {args.rows * args.cols} > 64, extra cells will be zero.")

    # 1) Start streamer
    streamer, smi = emagerv3_streamer(baud_rate=args.baud)
    print("[INFO] Streamer started.")
    print(f"[INFO] shared_memory_items tags: {[x[0] for x in smi]}")

    # 2) Create shared memory handler
    odh = OnlineDataHandler(shared_memory_items=smi)

    # 3) App
    app = QtWidgets.QApplication([])

    # 4) Rolling buffer
    rbuf = RollingBuffer(channels=64, width=args.plot_window, dtype=np.float32)

    # 5) Start consumer first so calibration has live data
    consumer = SharedMemoryConsumerVStack(odh=odh, rbuf=rbuf, poll_hz=args.poll_hz)
    consumer.start()

    # 6) Calibration or fallback defaults
    if args.no_calibration:
        max_intensity = args.default_max_intensity
        noise_floor = args.default_noise_floor
        print("[INFO] Calibration skipped.")
        print(f"[INFO] Using max_intensity={max_intensity}, noise_floor={noise_floor}")
    else:
        calib_dialog = CalibrationDialog(
            rbuf=rbuf,
            intensity_window=args.intensity_window,
            flex_duration=1.5,
            ext_duration=1.5,
            noise_duration=1.5
        )
        result = calib_dialog.exec_()
        if result != QtWidgets.QDialog.Accepted:
            print("[CALIB] Calibration canceled. Exiting.")
            try:
                consumer.stop()
                consumer.wait(1000)
            except Exception:
                pass
            try:
                streamer.stop()
            except Exception:
                try:
                    streamer.terminate()
                except Exception:
                    pass
            return

        max_flex = calib_dialog.max_flex
        max_ext = calib_dialog.max_ext
        noise_floor = calib_dialog.noise_floor

        max_intensity = max(max_flex, max_ext)
        if max_intensity <= 0:
            print("[CALIB] Warning: max_intensity <= 0; using fallback 1.0")
            max_intensity = 1.0

        if noise_floor >= max_intensity:
            print("[CALIB] Warning: noise_floor >= max_intensity; adjusting.")
            noise_floor = 0.5 * max_intensity

        print("\n[CALIB] Results:")
        print(f"  Max flexion intensity:   {max_flex:.3f}")
        print(f"  Max extension intensity: {max_ext:.3f}")
        print(f"  Final max_intensity:     {max_intensity:.3f}")
        print(f"  Noise floor:             {noise_floor:.3f}\n")

    # 7) Heatmap window
    win = LiveIntensityMap(
        fps=args.plot_fps,
        rows=args.rows,
        cols=args.cols,
        cmap_name=args.cmap,
        title=f"Live {args.rows}x{args.cols} EMG Heatmap [EMaGer v3 shared memory]",
        max_intensity=max_intensity
    )
    win.show()

    consumer.status.connect(
        lambda s: win.setWindowTitle(
            f"Live {args.rows}x{args.cols} EMG Heatmap [EMaGer v3 shared memory] | {s}"
        )
    )

    # 8) Heatmap update worker
    updater = HeatmapUpdateWorker(
        rbuf=rbuf,
        fps=args.plot_fps,
        intensity_window=args.intensity_window,
        rows=args.rows,
        cols=args.cols,
        noise_floor=noise_floor
    )
    updater.newImage.connect(win.update_from_intensity, QtCore.Qt.QueuedConnection)
    updater.start()

    # 9) Optional duration stop
    stop_timer = None
    if args.duration and args.duration > 0:
        def stop_after():
            win.close()

        stop_timer = QtCore.QTimer()
        stop_timer.setSingleShot(True)
        stop_timer.timeout.connect(stop_after)
        stop_timer.start(int(args.duration * 1000))

    # 10) Run + cleanup
    ret = 0
    try:
        ret = app.exec_()
    finally:
        try:
            updater.stop()
            updater.wait(1000)
        except Exception:
            pass

        try:
            consumer.stop()
            consumer.wait(1000)
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