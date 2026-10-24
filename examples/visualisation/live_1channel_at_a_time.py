#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live single-channel plotting using PyQtGraph, with data coming from LibEMG shared memory
via your EmagerV3Streamer.

Changes compared to the 64-channel grid version:
- Only one channel is shown at a time.
- A dropdown lets you select the channel.
- The y-axis is labelled.
"""

import time
import argparse
import numpy as np

# Import libemg before Qt on Windows
from libemg.data_handler import OnlineDataHandler
from libemg.streamers import emagerv3_streamer

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QFont
import pyqtgraph as pg


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
    def __init__(self, channels=64, width=4096, dtype=np.float32):
        self.channels = int(channels)
        self.width = int(width)
        self.buf = np.zeros((self.channels, self.width), dtype=dtype)
        self.write_idx = 0
        self.lock = QtCore.QMutex()
        self.valid = False

    def append_block(self, block_ch_time: np.ndarray):
        """
        Append a block shaped as:

            channels x time

        Example:
            64 x N
        """
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
        """
        Return a copy of the buffer in chronological order.

        Output shape:

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

        label = QtWidgets.QLabel("Choose the amplitude scale:")
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
        self.resize(500, 350)

    def _choose(self, y0, y1):
        self.y_min = y0
        self.y_max = y1
        self.accept()


# -----------------------------
# GUI single-channel viewer
# -----------------------------
# -----------------------------
# GUI single-channel viewer
# -----------------------------
class LiveSingleChannel(QtWidgets.QMainWindow):
    def __init__(
        self,
        width,
        fps=30.0,
        title="Live single-channel scope",
        y_min=0.0,
        y_max=4092.0,
        num_channels=64
    ):
        super().__init__()

        self.fps = float(fps)
        self.num_channels = int(num_channels)
        self.selected_channel = 0

        self.setWindowTitle(title)
        self.resize(1200, 700)

        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # -----------------------------
        # Min / Max tracking
        # -----------------------------
        self.running_min = None
        self.running_max = None

        pg.setConfigOptions(
            antialias=False,
            useOpenGL=True,
            enableExperimental=True
        )

        # Main Qt widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # -----------------------------
        # Top control row
        # -----------------------------
        control_layout = QtWidgets.QHBoxLayout()

        dropdown_label = QtWidgets.QLabel("Select channel:")
        dropdown_label.setFont(QFont("", 12))

        self.channel_dropdown = QtWidgets.QComboBox()
        self.channel_dropdown.setFont(QFont("", 12))
        self.channel_dropdown.setMinimumWidth(180)

        for ch in range(self.num_channels):
            self.channel_dropdown.addItem(f"Channel {ch}", ch)

        self.channel_dropdown.currentIndexChanged.connect(self._on_channel_changed)

        control_layout.addWidget(dropdown_label)
        control_layout.addWidget(self.channel_dropdown)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # -----------------------------
        # Plot
        # -----------------------------
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget)

        self.x = np.arange(width, dtype=float)

        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()

        self.plot_widget.setDownsampling(mode="peak", auto=True)
        self.plot_widget.setClipToView(True)

        self.plot_widget.setXRange(0, width - 1, padding=0)
        self.plot_widget.setYRange(self.y_min, self.y_max)

        # Axis labels
        self.plot_widget.setLabel("left", "Amplitude", units="ADC counts")
        self.plot_widget.setLabel("bottom", "Samples")

        self.curve = self.plot_widget.plot(pen=pg.mkPen("r", width=1))

        # -----------------------------
        # Min / Max text inside graph
        # -----------------------------
        self.minmax_text = pg.TextItem(
            text="Min: --   Max: --",
            color="w",
            anchor=(0, 0)
        )

        self.minmax_text.setFont(QFont("", 12))
        self.plot_widget.addItem(self.minmax_text)

        # Put text near the top-left of the visible graph
        self._update_minmax_text_position()

    def _update_minmax_text_position(self):
        """
        Place the Min/Max label near the top-left of the plot.
        """
        x_pos = 20
        y_pos = self.y_max - 0.08 * (self.y_max - self.y_min)
        self.minmax_text.setPos(x_pos, y_pos)

    def _reset_minmax(self):
        """
        Reset Min/Max values when changing channel.
        """
        self.running_min = None
        self.running_max = None
        self.minmax_text.setText("Min: --   Max: --")

    def _on_channel_changed(self, index):
        self.selected_channel = int(self.channel_dropdown.itemData(index))

        # Reset min/max when changing channel
        self._reset_minmax()

        self.setWindowTitle(
            f"Live single-channel scope | Channel {self.selected_channel}"
        )

    @QtCore.pyqtSlot(object)
    def update_from_block(self, view):
        """
        Receive a NumPy array shaped as:

            channels x width

        and draw only the selected channel.
        """
        if view is None:
            return

        ch = self.selected_channel

        if ch < 0 or ch >= view.shape[0]:
            return

        signal = view[ch]

        self.curve.setData(self.x, signal, _callSync="off")

        # -----------------------------
        # Update Min / Max values
        # -----------------------------
        if signal is not None and signal.size > 0:
            current_min = float(np.min(signal))
            current_max = float(np.max(signal))

            if self.running_min is None or current_min < self.running_min:
                self.running_min = current_min

            if self.running_max is None or current_max > self.running_max:
                self.running_max = current_max

            self.minmax_text.setText(
                f"Min: {self.running_min:.1f}   Max: {self.running_max:.1f}"
            )

            self._update_minmax_text_position()

# -----------------------------
# Consumer thread
# -----------------------------
class SharedMemoryConsumerVStack(QtCore.QThread):
    """
    Pulls from OnlineDataHandler shared memory and appends new EMG samples
    to the RollingBuffer.

    This is adapted to the vstack-prepend buffer behavior:

        newest rows are at the top of the shared memory buffer

    The streamer writes with something like:

        vstack((new[::-1], old_buffer))

    Therefore:
    - buffer[0] is the newest sample
    - buffer[new_rows - 1] is the oldest sample of the new chunk
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

    @staticmethod
    def _take_new_from_vstack_buffer(buf: np.ndarray, new_rows: int) -> np.ndarray:
        """
        Take only the new rows from a vstack-prepend buffer.

        Shared-memory buffer order:
            newest -> oldest

        Returned order:
            oldest -> newest
        """
        if new_rows <= 0:
            return np.zeros((0, buf.shape[1]), dtype=buf.dtype)

        new_rows = min(int(new_rows), int(buf.shape[0]))

        chunk_rev = buf[:new_rows, :]       # newest -> oldest
        chunk = chunk_rev[::-1, :].copy()   # oldest -> newest

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

            # First read: sync to current count so old backlog is ignored
            if not initialized:
                if MOD_EMG in count:
                    self.last_emg_count = int(count[MOD_EMG][0][0])

                if MOD_IMU in count:
                    self.last_imu_count = int(count[MOD_IMU][0][0])

                if MOD_SAMPLE_ID in count:
                    self.last_sid_count = int(count[MOD_SAMPLE_ID][0][0])

                initialized = True
                t_next = time.monotonic() + self.poll_period
                continue

            # -----------------------------
            # EMG handling
            # -----------------------------
            if MOD_EMG in vals and MOD_EMG in count:
                total_emg = int(count[MOD_EMG][0][0])
                new_emg = total_emg - self.last_emg_count

                if new_emg > 0:
                    emg_buf = vals[MOD_EMG]  # expected shape: time x channels

                    new_chunk = self._take_new_from_vstack_buffer(
                        emg_buf,
                        new_emg
                    )

                    if new_chunk is not None and new_chunk.size > 0:
                        # Convert from:
                        #   time x channels
                        # to:
                        #   channels x time
                        block_ch_time = new_chunk.T

                        self.rbuf.append_block(block_ch_time)

                    self.last_emg_count = total_emg

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
        description="PyQtGraph single-channel live view using LibEMG shared memory."
    )

    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds. Use 0 to run forever."
    )

    ap.add_argument(
        "--plot-window",
        type=int,
        default=6144,
        help="Samples kept on screen."
    )

    ap.add_argument(
        "--plot-fps",
        type=float,
        default=20.0,
        help="Plot update rate in Hz."
    )

    ap.add_argument(
        "--poll-hz",
        type=float,
        default=300.0,
        help="Shared-memory polling rate in Hz."
    )

    ap.add_argument(
        "--baud",
        type=int,
        default=3000000,
        help="Baud rate passed to emagerv3_streamer."
    )

    ap.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Number of EMG channels."
    )

    args = ap.parse_args()

    # -----------------------------
    # Start streamer
    # -----------------------------
    streamer, smi = emagerv3_streamer(baud_rate=args.baud)

    print("[INFO] Streamer started.")
    print(f"[INFO] shared_memory_items tags: {[x[0] for x in smi]}")

    # -----------------------------
    # Create OnlineDataHandler
    # -----------------------------
    odh = OnlineDataHandler(shared_memory_items=smi)

    # -----------------------------
    # Create Qt app
    # -----------------------------
    app = QtWidgets.QApplication([])

    # -----------------------------
    # Choose Y-axis scale
    # -----------------------------
    scale_dialog = ScaleSelectionDialog()
    result = scale_dialog.exec_()

    if result != QtWidgets.QDialog.Accepted:
        y_min = 0.0
        y_max = 4092.0
    else:
        y_min = float(scale_dialog.y_min)
        y_max = float(scale_dialog.y_max)

    print(f"[INFO] Using Y-range: {y_min} to {y_max}")

    # -----------------------------
    # Rolling buffer
    # -----------------------------
    rbuf = RollingBuffer(
        channels=args.channels,
        width=args.plot_window,
        dtype=np.float32
    )

    # -----------------------------
    # GUI window
    # -----------------------------
    win = LiveSingleChannel(
        width=args.plot_window,
        fps=args.plot_fps,
        title="Live single-channel scope [EMaGer v3 streamer | vstack buffer]",
        y_min=y_min,
        y_max=y_max,
        num_channels=args.channels
    )

    win.show()

    # -----------------------------
    # Update worker
    # -----------------------------
    updater = UpdateWorker(
        rbuf=rbuf,
        fps=args.plot_fps
    )

    updater.newBlock.connect(
        win.update_from_block,
        QtCore.Qt.QueuedConnection
    )

    updater.start()

    # -----------------------------
    # Shared-memory consumer
    # -----------------------------
    consumer = SharedMemoryConsumerVStack(
        odh=odh,
        rbuf=rbuf,
        poll_hz=args.poll_hz
    )

    consumer.status.connect(
        lambda s: win.setWindowTitle(
            f"Live single-channel scope | {s}"
        )
    )

    consumer.start()

    # -----------------------------
    # Optional duration stop
    # -----------------------------
    stop_timer = None

    if args.duration and args.duration > 0:
        def stop_after():
            win.close()

        stop_timer = QtCore.QTimer()
        stop_timer.setSingleShot(True)
        stop_timer.timeout.connect(stop_after)
        stop_timer.start(int(args.duration * 1000))

    # -----------------------------
    # Run app and cleanup
    # -----------------------------
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