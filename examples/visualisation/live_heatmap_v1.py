#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live 64-channel EMG heatmap for EMaGer V1 using HDSensor serial reader.

This version is adapted from the EMaGer V3 heatmap script, but instead of using:
    emagerv3_streamer -> shared memory -> OnlineDataHandler

it uses the same HDSensor / serial reading class as the V1 real-time oscilloscope.

Requirements:
    pip install numpy scipy pyserial pyqt6 pyqtgraph
"""

import time
import argparse
import logging

import numpy as np
import serial
import serial.tools.list_ports
import pyqtgraph as pg

from scipy import signal

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QFont


logger = logging.getLogger(__name__)


# -----------------------------
# Display scaling tweak
# -----------------------------
DISPLAY_SCALE_FACTOR = 0.85


# -----------------------------
# V1 packet reorder helper
# -----------------------------
def reorder(data, mask, match_result):
    """
    Looks for mask/template matching in data array and reorders.

    Parameters
    ----------
    data : np.ndarray
        1D uint8 data input.
    mask : np.ndarray
        1D mask to be matched.
    match_result : int
        Expected result of mask-data convolution matching.

    Returns
    -------
    list[np.ndarray] or None
        List of reordered 128-byte packets.
    """
    number_of_packet = int(len(data) / 128)
    roll_data = []

    for i in range(number_of_packet):
        packet = data[i * 128:(i + 1) * 128]
        data_lsb = packet & np.ones(128, dtype=np.uint8)

        mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), mode="valid")

        try:
            offset = np.where(mask_match == match_result)[0][0] - 3
        except IndexError:
            return None

        roll_data.append(np.roll(packet, -offset))

    return roll_data


# -----------------------------
# HDSensor serial reader
# -----------------------------
class HDSensor:
    """
    Sensor object for data logging from HD EMG sensor.

    This is the same style as the previous V1 real-time oscilloscope code.
    """

    def __init__(self, serialpath, BR):
        logger.info("init sensor")

        # Non-blocking serial port.
        # This helps avoid freezing the GUI.
        self.ser = serial.Serial(serialpath, BR, timeout=0)
        self.ser.close()

        self.bytes_to_read = 128

        # Template mask for packet alignment
        self.mask = np.array([0, 2] + [0, 1] * 63)

        # V1 channel map
        self.channelMap = np.array(
            [44, 49, 43, 55, 39, 59, 33, 2, 32, 3, 26, 6, 22, 13, 16, 10] +
            [42, 48, 45, 54, 38, 58, 35, 0, 34, 1, 27, 7, 23, 11, 17, 12] +
            [46, 52, 40, 51, 36, 56, 31, 60, 30, 63, 25, 4, 21, 8, 18, 15] +
            [47, 50, 41, 53, 37, 57, 29, 62, 28, 61, 24, 5, 19, 9, 20, 14],
            dtype=np.int64
        )

    def clear_buffer(self):
        self.ser.reset_input_buffer()

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def open(self):
        if not self.ser.is_open:
            self.ser.open()

    def live_read_fast(self, firstTime=False):
        """
        Fast non-blocking read for real-time plotting.

        Returns
        -------
        data_remap : np.ndarray or None
            Shape is 64 x nb_pts.
        nb_pts : int
            Number of new samples per channel.
        """

        if firstTime:
            self.open()
            self.clear_buffer()
            time.sleep(0.001)

        bytes_available = self.ser.in_waiting

        if bytes_available < 128:
            return None, 0

        bytes_to_read = bytes_available - (bytes_available % 128)

        if bytes_to_read <= 0:
            return None, 0

        raw = self.ser.read(bytes_to_read)

        if len(raw) < 128:
            return None, 0

        raw = np.frombuffer(raw, dtype=np.uint8)

        num_packets = len(raw) // 128
        raw = raw[:num_packets * 128].reshape(num_packets, 128)

        valid_packets = []

        for packet in raw:
            data_lsb = packet & 1
            mask_match = np.convolve(self.mask, np.r_[data_lsb, data_lsb], mode="valid")

            idx = np.where(mask_match == 63)[0]

            if len(idx) == 0:
                continue

            offset = idx[0] - 3
            packet = np.roll(packet, -offset)
            valid_packets.append(packet)

        if len(valid_packets) == 0:
            return None, 0

        packets = np.asarray(valid_packets, dtype=np.uint8)

        # Convert each 128-byte packet into 64 signed big-endian int16 samples.
        samples = packets.reshape(-1, 128).view(">i2").reshape(-1, 64)

        # Remap channels.
        samples = samples[:, self.channelMap]

        # Return shape: 64 x N
        data_remap = samples.T.astype(np.float32, copy=False)

        return data_remap, data_remap.shape[1]


# -----------------------------
# Port finder
# -----------------------------
def find_port(vid, pid):
    ports = serial.tools.list_ports.comports()

    for port in ports:
        if port.vid == vid and port.pid == pid:
            logger.info(f"Found device: {port.device}")
            return port.device

    raise ValueError("Device not found")


# -----------------------------
# Rolling buffer
# -----------------------------
class RollingBuffer:
    """
    Rolling buffer with shape channels x time.
    """

    def __init__(self, channels=64, width=4096, dtype=np.float32):
        self.channels = int(channels)
        self.width = int(width)
        self.buf = np.zeros((self.channels, self.width), dtype=dtype)
        self.write_idx = 0
        self.lock = QtCore.QMutex()
        self.valid = False

    def append_block(self, block_ch_time: np.ndarray):
        """
        Append a channels x N block.
        """
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
        """
        Return a copy of the rolling buffer in chronological order.

        Shape:
            channels x width
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
        lut = cmap.getLookupTable(0.0, 1.0, n)[:, :3].astype(np.ubyte)
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
    Compute intensity per channel.

    view:
        channels x width

    intensity:
        mean(abs(x - mean(x))) over last N samples for each channel.
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
# Serial reader thread
# -----------------------------
class SerialReaderWorker(QtCore.QThread):
    """
    Reads data from HDSensor in a background thread and pushes it into RollingBuffer.
    """

    status = QtCore.pyqtSignal(str)

    def __init__(self, sensor: HDSensor, rbuf: RollingBuffer, poll_hz=300.0, parent=None):
        super().__init__(parent)

        self.sensor = sensor
        self.rbuf = rbuf
        self.poll_period = 1.0 / max(1.0, float(poll_hz))
        self._stop = False

        self.first_go = True
        self.total_samples = 0
        self.last_status_time = 0.0

    def run(self):
        next_t = time.monotonic()

        while not self._stop:
            now = time.monotonic()

            if now < next_t:
                time.sleep(min(0.001, next_t - now))
                continue

            try:
                block_ch_time, nb_pts = self.sensor.live_read_fast(firstTime=self.first_go)
                self.first_go = False

                if block_ch_time is not None and nb_pts > 0:
                    self.rbuf.append_block(block_ch_time)
                    self.total_samples += nb_pts

                    now2 = time.monotonic()
                    if now2 - self.last_status_time > 0.5:
                        self.status.emit(f"samples={self.total_samples}")
                        self.last_status_time = now2

            except Exception as e:
                self.status.emit(f"Serial error: {e}")
                time.sleep(0.01)

            next_t = time.monotonic() + self.poll_period

    def stop(self):
        self._stop = True


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
        self.phase_names = ["Max FLEXION", "Max EXTENSION", "Relaxed noise"]
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
            f"{extra}\n"
            f"Duration: {duration:.1f} s"
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
                        self.max_flex = max(self.max_flex, val)
                    else:
                        self.max_ext = max(self.max_ext, val)

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
        rows=4,
        cols=16,
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
        self.resize(1000, 650)

        pg.setConfigOptions(
            antialias=False,
            useOpenGL=True
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
                self.img_item.setLevels((0.0, 1000.0))

            self.levels_set = True


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
        rows=4,
        cols=16,
        noise_floor: float = 0.0,
        transpose=True,
        parent=None
    ):
        super().__init__(parent)

        self.rbuf = rbuf
        self.period = 1.0 / max(1.0, float(fps))
        self.N = int(max(1, intensity_window))
        self.rows = int(rows)
        self.cols = int(cols)
        self.noise_floor = float(noise_floor)
        self.transpose = bool(transpose)
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

                    img = intens.reshape(self.rows, self.cols).astype(np.float32)

                    if self.transpose:
                        img = img.T

            self.newImage.emit(img)
            next_t += self.period

    def stop(self):
        self._stop = True


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Live 64-channel EMG heatmap using V1 HDSensor serial reader"
    )

    # Serial options
    parser.add_argument("--vid", type=lambda x: int(x, 0), default=0x04B4, help="USB VID")
    parser.add_argument("--pid", type=lambda x: int(x, 0), default=0xF155, help="USB PID")
    parser.add_argument("--port", type=str, default=None, help="Serial port. If omitted, VID/PID search is used.")
    parser.add_argument("--baud", type=int, default=1500000, help="Serial baud rate")

    # Runtime
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds. 0 = run forever.")

    # Buffer / rates
    parser.add_argument("--plot-window", type=int, default=4096, help="Samples per channel kept in rolling buffer")
    parser.add_argument("--plot-fps", type=float, default=30.0, help="Heatmap update rate in Hz")
    parser.add_argument("--poll-hz", type=float, default=300.0, help="Serial polling rate in Hz")
    parser.add_argument("--intensity-window", type=int, default=400, help="Samples used per channel for intensity")

    # Heatmap layout
    parser.add_argument("--rows", type=int, default=4, help="Heatmap rows")
    parser.add_argument("--cols", type=int, default=16, help="Heatmap columns")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap name")
    parser.add_argument("--no-transpose", action="store_true", help="Do not transpose the heatmap display")

    # Calibration
    parser.add_argument("--no-calibration", action="store_true", help="Skip flex/ext/noise calibration")
    parser.add_argument("--default-noise-floor", type=float, default=5.0, help="Noise floor if calibration is skipped")
    parser.add_argument(
        "--default-max-intensity",
        type=float,
        default=None,
        help="Optional fixed max intensity if calibration is skipped"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.rows * args.cols < 64:
        print(f"[WARN] rows*cols = {args.rows * args.cols} < 64, so not all channels will be shown.")
    elif args.rows * args.cols > 64:
        print(f"[WARN] rows*cols = {args.rows * args.cols} > 64, extra cells will be zero.")

    # Find serial port
    if args.port is None:
        port = find_port(args.vid, args.pid)
    else:
        port = args.port

    print(f"[INFO] Using serial port: {port}")
    print(f"[INFO] Baud rate: {args.baud}")

    sensor = HDSensor(port, args.baud)

    app = QtWidgets.QApplication([])

    # Rolling buffer
    rbuf = RollingBuffer(
        channels=64,
        width=args.plot_window,
        dtype=np.float32
    )

    # Start serial reader thread
    reader = SerialReaderWorker(
        sensor=sensor,
        rbuf=rbuf,
        poll_hz=args.poll_hz
    )
    reader.start()

    # Calibration or fallback
    if args.no_calibration:
        max_intensity = args.default_max_intensity
        noise_floor = args.default_noise_floor

        print("[INFO] Calibration skipped.")
        print(f"[INFO] Using max_intensity={max_intensity}, noise_floor={noise_floor}")

    else:
        print("[INFO] Waiting briefly for data before calibration...")
        time.sleep(0.5)

        calib_dialog = CalibrationDialog(
            rbuf=rbuf,
            intensity_window=args.intensity_window,
            flex_duration=1.5,
            ext_duration=1.5,
            noise_duration=1.5
        )

        result = calib_dialog.exec()

        if result != QtWidgets.QDialog.DialogCode.Accepted:
            print("[CALIB] Calibration canceled. Exiting.")

            try:
                reader.stop()
                reader.wait(1000)
            except Exception:
                pass

            try:
                sensor.close()
            except Exception:
                pass

            return 0

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

    # Heatmap window
    win = LiveIntensityMap(
        fps=args.plot_fps,
        rows=args.rows,
        cols=args.cols,
        cmap_name=args.cmap,
        title=f"Live {args.rows}x{args.cols} EMG Heatmap [V1 HDSensor]",
        max_intensity=max_intensity
    )

    win.show()

    reader.status.connect(
        lambda s: win.setWindowTitle(
            f"Live {args.rows}x{args.cols} EMG Heatmap [V1 HDSensor] | {s}"
        )
    )

    # Heatmap updater
    updater = HeatmapUpdateWorker(
        rbuf=rbuf,
        fps=args.plot_fps,
        intensity_window=args.intensity_window,
        rows=args.rows,
        cols=args.cols,
        noise_floor=noise_floor,
        transpose=not args.no_transpose
    )

    updater.newImage.connect(
        win.update_from_intensity,
        QtCore.Qt.ConnectionType.QueuedConnection
    )

    updater.start()

    # Optional duration stop
    stop_timer = None

    if args.duration and args.duration > 0:
        def stop_after():
            win.close()

        stop_timer = QtCore.QTimer()
        stop_timer.setSingleShot(True)
        stop_timer.timeout.connect(stop_after)
        stop_timer.start(int(args.duration * 1000))

    # Run + cleanup
    ret = 0

    try:
        ret = app.exec()

    finally:
        try:
            updater.stop()
            updater.wait(1000)
        except Exception:
            pass

        try:
            reader.stop()
            reader.wait(1000)
        except Exception:
            pass

        try:
            sensor.close()
        except Exception:
            pass

    return ret


if __name__ == "__main__":
    main()