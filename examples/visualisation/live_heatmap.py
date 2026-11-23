#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 64-channel EMG heatmap (muscle activation), no calibration needed.

Same heatmap as live_feature_projection.py:
- Data comes from the EMaGer streamer (version from the config) -> shared memory -> OnlineDataHandler
- Every frame (~30 Hz): take the latest samples of all channels, band-pass 20-450 Hz + notch at
  NOTCH_FREQ of the config (50/60 Hz)
- Value of each electrode = RMS over the last 200 ms
- Colour scale adapts by itself: it follows the strongest activation immediately (98th percentile of
  the electrodes) and then slowly comes back down (x0.99 per frame), so no calibration is needed.
  Yellow is reached at SCALE_TOP (70%) of that level, so medium activations stand out more
- Layout: 4 rows (rings) x 16 columns (electrodes around the arm), channel = ring * 16 + electrode

Keys:
    Esc     close the window
"""

import time
import logging
from pathlib import Path

import numpy as np
from scipy import signal

# Import libemg (and its transitive torch dep) BEFORE any Qt binding.
# On Windows, loading torch after a Qt binding produces WinError 1114.
from libemg.data_handler import OnlineDataHandler

from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.utils.streamer_utils import get_emager_streamer
from emagerlib.utils.notch import get_notch_freq

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# -----------------------------
# Parameters (same as live_feature_projection.py)
# -----------------------------
FPS = 30                 # display rate (Hz)
FEATURE_WINDOW_S = 0.2   # RMS window (s)
FILTER_PAD_S = 0.1       # extra samples used to settle the filters (s)
HEATMAP_ROWS, HEATMAP_COLS = 4, 16
SCALE_PERCENTILE = 98    # the colour scale follows this percentile of the electrodes...
SCALE_RELEASE = 0.99     # ...and comes back down by this factor per frame when the activation drops
SCALE_TOP = 0.7          # top of the colour scale (yellow) = this fraction of that level: the strongest
                         # electrodes saturate a little and medium activations look stronger (1.0 = no saturation)

logger = logging.getLogger(__name__)


def make_filter(fs, notch_freq):
    """Band-pass 20-450 Hz (4th order) + notch at notch_freq (Hz, bandwidth 3 Hz), as second-order sections."""
    high = min(450.0, 0.45 * fs)
    sos_bp = signal.butter(4, [20.0, high], btype="bandpass", fs=fs, output="sos")
    b, a = signal.iirnotch(notch_freq, Q=notch_freq / 3.0, fs=fs)
    return np.vstack([sos_bp, signal.tf2sos(b, a)])


def compute_rms(window, sos, rms_len):
    """window: (n_samples, n_channels) in chronological order.
    Returns the RMS of the last rms_len filtered samples of each channel."""
    filtered = signal.sosfiltfilt(sos, window, axis=0)[-rms_len:]
    return np.sqrt(np.mean(filtered ** 2, axis=0))


class LiveHeatmapWindow(QtWidgets.QMainWindow):
    def __init__(self, odh, fs, notch_freq):
        super().__init__()
        self.odh = odh
        self.rms_len = int(FEATURE_WINDOW_S * fs)
        self.n_samples = self.rms_len + int(FILTER_PAD_S * fs)
        self.notch_freq = notch_freq
        self.sos = make_filter(fs, notch_freq)
        self.heatmap_max = None
        self.last_count, self.last_count_time, self.sample_rate = None, time.monotonic(), 0.0

        self._build_ui()
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=self.close)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / FPS))
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    # ----- UI -----
    def _build_ui(self):
        self.setWindowTitle("Live EMG heatmap")
        self.resize(1400, 500)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(graphics, stretch=1)
        self.status = QtWidgets.QLabel()
        font = QtGui.QFont()
        font.setPointSize(12)
        self.status.setFont(font)
        layout.addWidget(self.status)
        self.setCentralWidget(central)

        self.heatmap_plot = graphics.addPlot(row=0, col=0, title="Muscle activation (RMS)")
        self.heatmap_plot.setMenuEnabled(False)
        self.heatmap_plot.setMouseEnabled(x=False, y=False)
        self.heatmap_plot.hideButtons()
        self.heatmap_plot.hideAxis("left")
        self.heatmap_plot.hideAxis("bottom")
        self.heatmap_plot.invertY(True)
        self.heatmap_plot.setAspectLocked(True)
        self.heatmap = pg.ImageItem()
        self.heatmap.setLookupTable(pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256))
        self.heatmap_plot.addItem(self.heatmap)

    # ----- Data -----
    def read_window(self):
        """Latest n_samples EMG samples in chronological order, or None if not enough data yet."""
        vals, count = self.odh.get_data(N=self.n_samples, filter=False)
        total = int(count["emg"][0][0])
        now = time.monotonic()
        if self.last_count is not None and now - self.last_count_time >= 1.0:
            self.sample_rate = (total - self.last_count) / (now - self.last_count_time)
            self.last_count, self.last_count_time = total, now
        elif self.last_count is None:
            self.last_count, self.last_count_time = total, now
        if total < self.n_samples:
            return None
        # Shared memory holds the newest sample first
        return np.asarray(vals["emg"][:self.n_samples][::-1], dtype=np.float64)

    def update_frame(self):
        window = self.read_window()
        if window is None:
            self.status.setText("Waiting for EMG data...")
            return
        self.update_heatmap(compute_rms(window, self.sos, self.rms_len))
        self.status.setText(
            f"{self.sample_rate:.0f} samples/s   |   notch {self.notch_freq} Hz   |   "
            f"colour scale 0 - {SCALE_TOP * self.heatmap_max:.1f} (adapts by itself)        [Esc] close")

    # ----- Drawing -----
    def update_heatmap(self, rms):
        n = HEATMAP_ROWS * HEATMAP_COLS
        values = np.zeros(n)
        values[:min(n, rms.size)] = rms[:n]
        # Slowly adapting colour scale (fast attack, slow release)
        peak = float(np.percentile(values, SCALE_PERCENTILE))
        self.heatmap_max = peak if self.heatmap_max is None else max(peak, SCALE_RELEASE * self.heatmap_max)
        self.heatmap.setImage(values.reshape(HEATMAP_ROWS, HEATMAP_COLS).T, autoLevels=False,
                              levels=(0.0, max(SCALE_TOP * self.heatmap_max, 1e-9)))


def parse_args(argv=None):
    parser = create_parser(
        description="Live 64-channel EMG heatmap (adaptive colour scale, no calibration)",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="live_heatmap")
    save_config_if_requested(args, cfg, script_name="live_heatmap")
    return args, cfg


def main(argv=None):
    _, cfg = setup_runtime(argv)

    p, smi = get_emager_streamer(cfg.EMAGER_VERSION)
    smi_emg = [item for item in smi if item[0] in ['emg', 'emg_count']]
    logger.info(f"Streamer created: process: {p}, smi : {smi_emg}")
    odh = OnlineDataHandler(shared_memory_items=smi_emg)
    logger.info("Data handler created")

    app = QtWidgets.QApplication([])
    win = LiveHeatmapWindow(odh, fs=cfg.SAMPLING, notch_freq=get_notch_freq(cfg))
    win.show()
    app.exec_()

    try:
        p.stop()
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
