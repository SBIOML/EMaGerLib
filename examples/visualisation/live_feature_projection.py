#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 2D projection of EMG muscle-activation features (demo tool).

Data comes from the EMaGer streamer (version from the config) -> shared memory -> OnlineDataHandler.

Pipeline (every frame, ~30 Hz):
- Take the latest samples of all channels, band-pass 20-450 Hz + notch 60 Hz
- Feature vector = log(RMS) of each channel over the last 200 ms (one value per electrode)
- Project the feature vector to 2D with an adaptive PCA

Adaptive PCA (no calibration needed):
- The last MEMORY_SECONDS of feature vectors are kept in memory
- PCA is refitted every REFIT_PERIOD seconds on that memory, so the map organises itself
  as the person performs different gestures
- Each new fit is rotated/flipped to best match the previous one (orthogonal Procrustes),
  so the map does not jump around between refits
- Press SPACE to freeze the projection once it looks good: each gesture then always lands
  in the same place

Keys:
    SPACE   freeze / unfreeze the projection
    R       reset (forget everything and start learning again)
    1-9     colour the points with this gesture label (optional, to show clusters)
    0       no label (grey)
"""

import time
import logging
from collections import deque
from pathlib import Path

import numpy as np
from scipy import signal

# Import libemg (and its transitive torch dep) BEFORE any Qt binding.
# On Windows, loading torch after a Qt binding produces WinError 1114.
from libemg.data_handler import OnlineDataHandler

from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.utils.streamer_utils import get_emager_streamer

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# -----------------------------
# Parameters
# -----------------------------
FPS = 30                 # display / feature rate (Hz)
FEATURE_WINDOW_S = 0.2   # RMS window (s)
FILTER_PAD_S = 0.1       # extra samples used to settle the filters (s)
MEMORY_SECONDS = 60      # how much recent history the PCA learns from (s)
MIN_POINTS_TO_FIT = 30   # ~1 s of data before the first projection
REFIT_PERIOD = 0.5       # PCA refit period in adaptive mode (s)
TRAIL_LENGTH = 45        # points in the moving trail (~1.5 s)
HEATMAP_ROWS, HEATMAP_COLS = 4, 16
LOG_EPS = 1e-3

# Gesture label colours (key 1-9), label 0 = unlabelled
LABEL_COLORS = [
    (150, 150, 150), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
]

logger = logging.getLogger(__name__)


def make_filter(fs):
    """Band-pass 20-450 Hz (4th order) + notch 60 Hz, as second-order sections."""
    high = min(450.0, 0.45 * fs)
    sos_bp = signal.butter(4, [20.0, high], btype="bandpass", fs=fs, output="sos")
    b, a = signal.iirnotch(60.0, Q=60.0 / 3.0, fs=fs)
    return np.vstack([sos_bp, signal.tf2sos(b, a)])


def compute_features(window, sos, rms_len):
    """window: (n_samples, n_channels) in chronological order.
    Returns (rms, log_rms) of the last rms_len filtered samples of each channel."""
    filtered = signal.sosfiltfilt(sos, window, axis=0)[-rms_len:]
    rms = np.sqrt(np.mean(filtered ** 2, axis=0))
    return rms, np.log(rms + LOG_EPS)


class AdaptiveProjection:
    """2D PCA refitted on a rolling memory of feature vectors, kept aligned between refits."""

    def __init__(self, memory_size):
        self.features = deque(maxlen=memory_size)
        self.labels = deque(maxlen=memory_size)
        self.mean = None
        self.components = None   # (2, n_features)
        self.explained = (0.0, 0.0)

    def reset(self):
        self.features.clear()
        self.labels.clear()
        self.mean = None
        self.components = None
        self.explained = (0.0, 0.0)

    @property
    def ready(self):
        return self.components is not None

    def add(self, feature, label):
        self.features.append(feature)
        self.labels.append(label)

    def refit(self):
        if len(self.features) < MIN_POINTS_TO_FIT:
            return
        X = np.asarray(self.features)
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        components = vt[:2]
        if self.components is not None:
            # Rotate/flip the new axes to best match the previous ones (orthogonal Procrustes)
            u, _, vt_r = np.linalg.svd(self.components @ components.T)
            components = (u @ vt_r) @ components
        variance = s ** 2
        self.explained = tuple(variance[:2] / max(variance.sum(), 1e-12))
        self.mean, self.components = mean, components

    def project(self, X):
        return (np.atleast_2d(X) - self.mean) @ self.components.T


class FeatureProjectionWindow(QtWidgets.QMainWindow):
    def __init__(self, odh, fs):
        super().__init__()
        self.odh = odh
        self.rms_len = int(FEATURE_WINDOW_S * fs)
        self.n_samples = self.rms_len + int(FILTER_PAD_S * fs)
        self.sos = make_filter(fs)

        self.projection = AdaptiveProjection(memory_size=MEMORY_SECONDS * FPS)
        self.trail = deque(maxlen=TRAIL_LENGTH)   # raw feature vectors, re-projected every frame
        self.frozen = False
        self.label = 0
        self.last_refit = 0.0
        self.view_range = None
        self.last_count, self.last_count_time, self.sample_rate = None, time.monotonic(), 0.0

        self.brushes = [pg.mkBrush(*c, 110) for c in LABEL_COLORS]
        self._build_ui()
        self._build_shortcuts()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / FPS))
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    # ----- UI -----
    def _build_ui(self):
        self.setWindowTitle("Live EMG feature projection")
        self.resize(1400, 800)
        pg.setConfigOptions(antialias=True)

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

        # Heatmap of the current muscle activation
        self.heatmap_plot = graphics.addPlot(row=0, col=0, title="Muscle activation (RMS)")
        self.heatmap_plot.setMenuEnabled(False)
        self.heatmap_plot.setMouseEnabled(x=False, y=False)
        self.heatmap_plot.hideAxis("left")
        self.heatmap_plot.hideAxis("bottom")
        self.heatmap_plot.invertY(True)
        self.heatmap_plot.setAspectLocked(True)
        self.heatmap = pg.ImageItem()
        self.heatmap.setLookupTable(pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256))
        self.heatmap_plot.addItem(self.heatmap)
        self.heatmap_max = None

        # 2D projection
        self.proj_plot = graphics.addPlot(row=0, col=1, title="Feature projection (adaptive PCA)")
        self.proj_plot.setMenuEnabled(False)
        self.proj_plot.setMouseEnabled(x=False, y=False)
        self.proj_plot.hideButtons()
        self.proj_plot.showGrid(x=True, y=True, alpha=0.2)
        graphics.ci.layout.setColumnStretchFactor(0, 1)
        graphics.ci.layout.setColumnStretchFactor(1, 3)

        self.memory_scatter = pg.ScatterPlotItem(size=6, pen=None)
        self.trail_curve = pg.PlotCurveItem(pen=pg.mkPen(255, 255, 255, 90, width=2))
        self.trail_scatter = pg.ScatterPlotItem(pen=None)
        self.current_point = pg.ScatterPlotItem(size=26, pen=pg.mkPen("w", width=3))
        for item in (self.memory_scatter, self.trail_curve, self.trail_scatter, self.current_point):
            self.proj_plot.addItem(item)

    def _build_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=self.toggle_freeze)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_R), self, activated=self.reset)
        for label in range(10):
            QtWidgets.QShortcut(QtGui.QKeySequence(str(label)), self,
                                activated=lambda l=label: self.set_label(l))

    # ----- Controls -----
    def toggle_freeze(self):
        if not self.projection.ready:
            return
        self.frozen = not self.frozen
        logger.info("Projection frozen" if self.frozen else "Projection adaptive")

    def reset(self):
        self.projection.reset()
        self.trail.clear()
        self.frozen = False
        self.view_range = None
        logger.info("Projection reset")

    def set_label(self, label):
        self.label = label

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

        rms, feature = compute_features(window, self.sos, self.rms_len)
        self.update_heatmap(rms)

        self.projection.add(feature, self.label)
        self.trail.append(feature)
        now = time.monotonic()
        if not self.frozen and now - self.last_refit >= REFIT_PERIOD:
            self.projection.refit()
            self.last_refit = now

        if self.projection.ready:
            self.update_projection()
        self.update_status()

    # ----- Drawing -----
    def update_heatmap(self, rms):
        n = HEATMAP_ROWS * HEATMAP_COLS
        values = np.zeros(n)
        values[:min(n, rms.size)] = rms[:n]
        # Slowly adapting colour scale (fast attack, slow release)
        peak = float(np.percentile(values, 98))
        self.heatmap_max = peak if self.heatmap_max is None else max(peak, 0.99 * self.heatmap_max)
        self.heatmap.setImage(values.reshape(HEATMAP_ROWS, HEATMAP_COLS).T, autoLevels=False,
                              levels=(0.0, max(self.heatmap_max, 1e-9)))

    def update_projection(self):
        memory = self.projection.project(np.asarray(self.projection.features))
        labels = list(self.projection.labels)
        self.memory_scatter.setData(pos=memory, brush=[self.brushes[l] for l in labels])

        trail = self.projection.project(np.asarray(self.trail))
        n = len(trail)
        alphas = np.linspace(40, 220, n).astype(int)
        sizes = np.linspace(4, 14, n)
        color = LABEL_COLORS[self.label] if self.label else (255, 255, 255)
        self.trail_curve.setData(trail[:, 0], trail[:, 1])
        self.trail_scatter.setData(pos=trail, size=sizes, brush=[pg.mkBrush(*color, a) for a in alphas])
        self.current_point.setData(pos=trail[-1:], brush=pg.mkBrush(*(LABEL_COLORS[self.label] if self.label else (255, 80, 80))))

        # Smoothly follow the extent of the points so the view does not jump
        lo = np.percentile(memory, 1, axis=0)
        hi = np.percentile(memory, 99, axis=0)
        margin = 0.15 * np.maximum(hi - lo, 1e-6)
        target = np.concatenate([lo - margin, hi + margin])
        self.view_range = target if self.view_range is None else 0.9 * self.view_range + 0.1 * target
        self.proj_plot.setXRange(self.view_range[0], self.view_range[2], padding=0)
        self.proj_plot.setYRange(self.view_range[1], self.view_range[3], padding=0)

        ev1, ev2 = self.projection.explained
        self.proj_plot.setLabel("bottom", f"PC1 ({100 * ev1:.0f}% of variance)")
        self.proj_plot.setLabel("left", f"PC2 ({100 * ev2:.0f}% of variance)")
        mode = "FROZEN" if self.frozen else "adaptive PCA"
        self.proj_plot.setTitle(f"Feature projection ({mode})")

    def update_status(self):
        if not self.projection.ready:
            mode = f"Learning... ({len(self.projection.features)}/{MIN_POINTS_TO_FIT} windows)"
        else:
            mode = "FROZEN" if self.frozen else "ADAPTIVE"
        label = f"label {self.label}" if self.label else "no label"
        self.status.setText(
            f"{mode}   |   {label}   |   {len(self.projection.features)} windows in memory   |   "
            f"{self.sample_rate:.0f} samples/s        "
            f"[SPACE] freeze/unfreeze   [R] reset   [1-9] label   [0] no label")


def parse_args(argv=None):
    parser = create_parser(
        description="Live 2D projection of EMG features (adaptive PCA)",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="live_feature_projection")
    save_config_if_requested(args, cfg, script_name="live_feature_projection")
    return args, cfg


def main(argv=None):
    _, cfg = setup_runtime(argv)

    p, smi = get_emager_streamer(cfg.EMAGER_VERSION)
    smi_emg = [item for item in smi if item[0] in ['emg', 'emg_count']]
    logger.info(f"Streamer created: process: {p}, smi : {smi_emg}")
    odh = OnlineDataHandler(shared_memory_items=smi_emg)
    logger.info("Data handler created")

    app = QtWidgets.QApplication([])
    win = FeatureProjectionWindow(odh, fs=cfg.SAMPLING)
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
