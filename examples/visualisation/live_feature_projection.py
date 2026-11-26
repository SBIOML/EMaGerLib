#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 2D projection of EMG muscle-activation features (demo tool).

Data comes from the EMaGer streamer (version from the config) -> shared memory -> OnlineDataHandler.

Pipeline (every frame, ~30 Hz):
- Take the latest samples of all channels, band-pass 20-450 Hz + notch at NOTCH_FREQ of the config (50/60 Hz)
- Feature vector = log(RMS) of each channel over the last 200 ms (one value per electrode)
- Project the feature vector to 2D with an adaptive PCA

Adaptive PCA (no calibration needed):
- Unlabelled (grey) feature vectors are kept for GREY_MEMORY_SECONDS, then forgotten
- Labelled (coloured) feature vectors are kept forever (until R)
- PCA is refitted every REFIT_PERIOD seconds on all the points kept (grey + coloured), so the map
  organises itself as the person performs different gestures
- Each new fit is rotated/flipped to best match the previous one (orthogonal Procrustes),
  so the map does not jump around between refits
- Press SPACE to freeze the projection once it looks good: each gesture then always lands
  in the same place

Keys:
    SPACE   freeze / unfreeze the projection
    R       reset (forget everything and start learning again)
    1-9, 0  record LABEL_SECONDS of points in the colour of that gesture (see GESTURES), then back
            to grey. Pressing a key again erases its previous points first. The panel on the right
            shows each key with its colour, gesture picture and whether it is used.
    Esc     stop recording now (back to grey)
    P       show / hide the gesture pictures at the centre of each gesture's points
            (same as the button above the panel)

Gesture pictures are read from MEDIA_PATH of the config (the folder where libEMG downloads them).
"""

import os
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
from emagerlib.utils.notch import get_notch_freq

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
GREY_MEMORY_SECONDS = 15 # unlabelled (grey) points are forgotten after this (s); coloured points are kept forever
LABEL_SECONDS = 5        # duration recorded in a colour after pressing 1-9 (s), then back to grey
MIN_POINTS_TO_FIT = 30   # ~1 s of data before the first projection
REFIT_PERIOD = 0.5       # PCA refit period in adaptive mode (s)
TRAIL_LENGTH = 45        # points in the moving trail (~1.5 s)
HEATMAP_ROWS, HEATMAP_COLS = 4, 16
LOG_EPS = 1e-3

# Gestures in key order: label 1 = key 1, ..., label 9 = key 9, label 10 = key 0.
# Names are the picture file names in MEDIA_PATH (without .png).
GESTURES = ["No_Motion", "Wrist_Extension", "Wrist_Flexion", "Hand_Close", "Hand_Open",
            "Thumbs_Up", "Index_Extension", "Index_Pinch", "Pronation", "Supination"]
PANEL_PICTURE_SIZE = 44  # gesture pictures in the side panel (pixels)
PLOT_PICTURE_SIZE = 64   # gesture pictures placed on the projection (pixels)

# Colour of each label, label 0 = unlabelled (grey)
LABEL_COLORS = [
    (150, 150, 150), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (170, 110, 40),
]
DEFAULT_MEDIA_PATH = Path(__file__).parent.parent.parent / "media-test"


def label_key(label):
    """Keyboard key of a label: 1-9, and 0 for label 10."""
    return str(label % 10)

logger = logging.getLogger(__name__)


def make_filter(fs, notch_freq):
    """Band-pass 20-450 Hz (4th order) + notch at notch_freq (Hz, bandwidth 3 Hz), as second-order sections."""
    high = min(450.0, 0.45 * fs)
    sos_bp = signal.butter(4, [20.0, high], btype="bandpass", fs=fs, output="sos")
    b, a = signal.iirnotch(notch_freq, Q=notch_freq / 3.0, fs=fs)
    return np.vstack([sos_bp, signal.tf2sos(b, a)])


def compute_features(window, sos, rms_len):
    """window: (n_samples, n_channels) in chronological order.
    Returns (rms, log_rms) of the last rms_len filtered samples of each channel."""
    filtered = signal.sosfiltfilt(sos, window, axis=0)[-rms_len:]
    rms = np.sqrt(np.mean(filtered ** 2, axis=0))
    return rms, np.log(rms + LOG_EPS)


class GrowingArray:
    """Rows appended one at a time, read as a numpy array without copying (capacity doubles when full)."""

    def __init__(self):
        self.data, self.size = None, 0

    def append(self, row):
        if self.data is None:
            self.data = np.empty((1024, len(row)))
        elif self.size == len(self.data):
            self.data = np.concatenate([self.data, np.empty_like(self.data)])
        self.data[self.size] = row
        self.size += 1

    def array(self):
        return self.data[:self.size] if self.data is not None else np.empty((0, 0))

    def __len__(self):
        return self.size


class AdaptiveProjection:
    """2D PCA refitted on the points kept in memory, kept aligned between refits.

    Grey (unlabelled) points: removed grey_seconds after they were added. Coloured points: kept forever."""

    def __init__(self, grey_seconds):
        self.grey_seconds = grey_seconds
        self.grey = deque()
        self.grey_times = deque()
        self.coloured = {}       # label -> GrowingArray of feature vectors
        self.mean = None
        self.components = None   # (2, n_features)
        self.explained = (0.0, 0.0)

    def reset(self):
        self.grey.clear()
        self.grey_times.clear()
        self.coloured = {}
        self.mean = None
        self.components = None
        self.explained = (0.0, 0.0)

    @property
    def ready(self):
        return self.components is not None

    @property
    def n_coloured(self):
        return sum(len(points) for points in self.coloured.values())

    def __len__(self):
        return len(self.grey) + self.n_coloured

    def add(self, feature, label):
        now = time.monotonic()
        if label == 0:
            self.grey.append(feature)
            self.grey_times.append(now)
        else:
            self.coloured.setdefault(label, GrowingArray()).append(feature)
        while self.grey_times and now - self.grey_times[0] > self.grey_seconds:
            self.grey.popleft()
            self.grey_times.popleft()

    def clear_label(self, label):
        """Forget the coloured points of one label."""
        self.coloured.pop(label, None)

    def all_points(self):
        blocks = [np.asarray(self.grey)] if self.grey else []
        blocks += [points.array() for points in self.coloured.values()]
        return np.concatenate(blocks) if blocks else np.empty((0, 0))

    def refit(self):
        if len(self) < MIN_POINTS_TO_FIT:
            return
        X = self.all_points()
        mean = X.mean(axis=0)
        # Eigen-decomposition of the 64 x 64 covariance: same axes as an SVD of the data, but its cost
        # does not grow with the number of points (coloured points accumulate forever)
        centered = X - mean
        variance, vectors = np.linalg.eigh(centered.T @ centered)
        order = np.argsort(variance)[::-1]
        variance, components = variance[order], vectors[:, order[:2]].T
        if self.components is not None:
            # Rotate/flip the new axes to best match the previous ones (orthogonal Procrustes)
            u, _, vt_r = np.linalg.svd(self.components @ components.T)
            components = (u @ vt_r) @ components
        self.explained = tuple(variance[:2] / max(variance.sum(), 1e-12))
        self.mean, self.components = mean, components

    def project(self, X):
        return (np.atleast_2d(X) - self.mean) @ self.components.T


class FeatureProjectionWindow(QtWidgets.QMainWindow):
    def __init__(self, odh, fs, notch_freq, media_path=None):
        super().__init__()
        self.odh = odh
        self.media_path = Path(media_path) if media_path else DEFAULT_MEDIA_PATH
        self.rms_len = int(FEATURE_WINDOW_S * fs)
        self.n_samples = self.rms_len + int(FILTER_PAD_S * fs)
        self.notch_freq = notch_freq
        self.sos = make_filter(fs, notch_freq)

        self.projection = AdaptiveProjection(grey_seconds=GREY_MEMORY_SECONDS)
        self.trail = deque(maxlen=TRAIL_LENGTH)   # raw feature vectors, re-projected every frame
        self.frozen = False
        self.label = 0
        self.label_until = 0.0   # time when the current coloured recording ends
        self.last_refit = 0.0
        self.view_range = None
        self.last_count, self.last_count_time, self.sample_rate = None, time.monotonic(), 0.0

        self.brushes = [pg.mkBrush(*c, 110) for c in LABEL_COLORS]
        self._build_ui()
        self._build_shortcuts()
        self.update_legend()

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
        top = QtWidgets.QHBoxLayout()
        graphics = pg.GraphicsLayoutWidget()
        top.addWidget(graphics, stretch=1)
        top.addWidget(self._build_legend())
        layout.addLayout(top, stretch=1)
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

        # One scatter per label (fixed colour): fast to redraw even with many coloured points
        self.memory_scatters = [pg.ScatterPlotItem(size=6, pen=None, brush=brush) for brush in self.brushes]
        self.trail_curve = pg.PlotCurveItem(pen=pg.mkPen(255, 255, 255, 90, width=2))
        self.trail_scatter = pg.ScatterPlotItem(pen=None)
        self.current_point = pg.ScatterPlotItem(size=26, pen=pg.mkPen("w", width=3))
        for item in (*self.memory_scatters, self.trail_curve, self.trail_scatter, self.current_point):
            self.proj_plot.addItem(item)
        # Drawing order: stored points (0) < gesture pictures (5) < newest points / trail (10)
        for item in (self.trail_curve, self.trail_scatter, self.current_point):
            item.setZValue(10)

        # Gesture pictures shown at the centre of each gesture's points (fixed size in pixels)
        self.plot_pictures = {}
        for label, pixmap in self.gesture_pixmaps(PLOT_PICTURE_SIZE, border=4).items():
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            item.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)
            item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            item.setZValue(5)
            item.setVisible(False)
            self.proj_plot.addItem(item, ignoreBounds=True)
            self.plot_pictures[label] = item

    def gesture_pixmaps(self, size, border=0):
        """{label: square picture of its gesture}, with a border in the label colour if border > 0."""
        pixmaps = {}
        for label, gesture in enumerate(GESTURES, start=1):
            file = self.media_path / f"{gesture}.png"
            if not file.exists():
                logger.warning(f"Gesture picture not found: {file}")
                continue
            inner = size - 2 * border
            picture = QtGui.QPixmap(str(file)).scaled(inner, inner, QtCore.Qt.KeepAspectRatio,
                                                      QtCore.Qt.SmoothTransformation)
            pixmap = QtGui.QPixmap(size, size)
            pixmap.fill(QtGui.QColor(*LABEL_COLORS[label]))
            painter = QtGui.QPainter(pixmap)
            painter.drawPixmap((size - picture.width()) // 2, (size - picture.height()) // 2, picture)
            painter.end()
            pixmaps[label] = pixmap
        return pixmaps

    def _build_legend(self):
        """Panel listing every key with its colour, gesture picture and whether it is used / recording."""
        panel = QtWidgets.QGroupBox("Gestures")
        panel.setFixedWidth(330)
        grid = QtWidgets.QGridLayout(panel)
        font = QtGui.QFont()
        font.setPointSize(11)

        self.pictures_button = QtWidgets.QPushButton("Show gesture pictures on the map  (P)")
        self.pictures_button.setCheckable(True)
        self.pictures_button.setFocusPolicy(QtCore.Qt.NoFocus)  # so SPACE keeps freezing, not clicking
        self.pictures_button.toggled.connect(self.on_pictures_toggled)
        grid.addWidget(self.pictures_button, 0, 0, 1, 4)

        pictures = self.gesture_pixmaps(PANEL_PICTURE_SIZE)
        self.legend_rows = {}
        for label, gesture in enumerate(GESTURES, start=1):
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(20, 20)
            swatch.setStyleSheet("background-color: rgb(%d, %d, %d); border-radius: 4px" % LABEL_COLORS[label])
            key = QtWidgets.QLabel(label_key(label))
            key.setFont(font)
            picture = QtWidgets.QLabel()
            picture.setFixedSize(PANEL_PICTURE_SIZE, PANEL_PICTURE_SIZE)
            if label in pictures:
                picture.setPixmap(pictures[label])
            text = QtWidgets.QLabel()
            text.setFont(font)
            grid.addWidget(swatch, label, 0)
            grid.addWidget(key, label, 1)
            grid.addWidget(picture, label, 2)
            grid.addWidget(text, label, 3)
            self.legend_rows[label] = (key, text)
        hint = QtWidgets.QLabel(f"Press a key to record {LABEL_SECONDS} s of that gesture "
                                "(erases its previous points). Esc = stop.")
        hint.setStyleSheet("color: gray")
        hint.setWordWrap(True)
        grid.addWidget(hint, len(GESTURES) + 1, 0, 1, 4)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(len(GESTURES) + 2, 1)
        return panel

    def on_pictures_toggled(self, shown):
        self.pictures_button.setText(("Hide" if shown else "Show") + " gesture pictures on the map  (P)")
        if self.projection.ready:
            self.update_projection()

    def update_legend(self):
        remaining = self.label_until - time.monotonic()
        for label, (key, state) in self.legend_rows.items():
            n = len(self.projection.coloured.get(label, ()))
            name = GESTURES[label - 1].replace("_", " ")
            if label == self.label:
                state.setText(f"{name}\nrecording {max(remaining, 0):.1f} s")
                weight = "bold"
            elif n:
                state.setText(f"{name}\nused ({n} points)")
                weight = "normal"
            else:
                state.setText(f"{name}\nunused")
                weight = "normal"
            color = "black" if n or label == self.label else "gray"
            for widget in (key, state):
                widget.setStyleSheet(f"color: {color}; font-weight: {weight}")

    def _build_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=self.toggle_freeze)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_R), self, activated=self.reset)
        for label in range(1, len(GESTURES) + 1):
            QtWidgets.QShortcut(QtGui.QKeySequence(label_key(label)), self,
                                activated=lambda l=label: self.set_label(l))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=lambda: self.set_label(0))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_P), self, activated=self.pictures_button.toggle)

    # ----- Controls -----
    def toggle_freeze(self):
        if not self.projection.ready:
            return
        self.frozen = not self.frozen
        logger.info("Projection frozen" if self.frozen else "Projection adaptive")

    def reset(self):
        self.projection.reset()
        self.trail.clear()
        self.label = 0
        self.update_legend()
        self.frozen = False
        self.view_range = None
        logger.info("Projection reset")

    def set_label(self, label):
        """1-9: erase that colour's points and record LABEL_SECONDS in it. 0: back to grey now."""
        if label:
            self.projection.clear_label(label)
            self.label_until = time.monotonic() + LABEL_SECONDS
        self.label = label
        self.update_legend()

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

        now = time.monotonic()
        if self.label and now >= self.label_until:
            self.label = 0   # coloured recording finished: back to grey
        self.projection.add(feature, self.label)
        self.trail.append(feature)
        if not self.frozen and now - self.last_refit >= REFIT_PERIOD:
            self.projection.refit()
            self.last_refit = now

        if self.projection.ready:
            self.update_projection()
        self.update_status()
        self.update_legend()

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
        groups = {0: np.asarray(self.projection.grey)}
        groups.update({label: points.array() for label, points in self.projection.coloured.items()})
        projected = []
        for label, scatter in enumerate(self.memory_scatters):
            points = groups.get(label)
            if points is not None and len(points):
                xy = self.projection.project(points)
                # colour, size and pen must be given again: setData resets them to the defaults
                scatter.setData(pos=xy, brush=self.brushes[label], size=6, pen=None)
                projected.append(xy)
            else:
                scatter.setData(pos=np.empty((0, 2)))
        memory = np.concatenate(projected)

        # Gesture pictures at the centre (median) of each gesture's points
        show = self.pictures_button.isChecked()
        for label, item in self.plot_pictures.items():
            points = groups.get(label)
            visible = show and points is not None and len(points) > 0
            if visible:
                x, y = np.median(self.projection.project(points), axis=0)
                item.setPos(float(x), float(y))
            item.setVisible(visible)

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
            mode = f"Learning... ({len(self.projection)}/{MIN_POINTS_TO_FIT} windows)"
        else:
            mode = "FROZEN" if self.frozen else "ADAPTIVE"
        label = f"label {self.label}" if self.label else "no label"
        self.status.setText(
            f"{mode}   |   {label}   |   {len(self.projection.grey)} grey (last {GREY_MEMORY_SECONDS} s) + "
            f"{self.projection.n_coloured} coloured windows   |   "
            f"{self.sample_rate:.0f} samples/s   |   notch {self.notch_freq} Hz        "
            f"[SPACE] freeze/unfreeze   [R] reset   [1-9, 0] record {LABEL_SECONDS} s of a gesture   [Esc] stop   [P] pictures")


def make_app():
    """QApplication with per-monitor high-DPI scaling and exact scale factors (e.g. 1.5).
    Without it (Qt 5 default), on a screen whose scale differs from the main one (e.g. laptop at 150%,
    external monitor at 100%), the points / heatmap are drawn at a different scale than their axes.
    Same settings as pyqtgraph's own pg.mkQApp(). Must run before the QApplication is created."""
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    return QtWidgets.QApplication([])


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

    app = make_app()
    win = FeatureProjectionWindow(odh, fs=cfg.SAMPLING, notch_freq=get_notch_freq(cfg),
                                  media_path=getattr(cfg, "MEDIA_PATH", None))
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
