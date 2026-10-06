import numpy as np
import libemg
import glob
import argparse
import os
from Backbone import Backbone

SAMPLING_RATE = 1000
WINDOW_MS = 200
WINDOW_INC_MS = 50
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_MS / 1000)
WINDOW_INC = int(SAMPLING_RATE * WINDOW_INC_MS / 1000)
FEATURE_LIST = ['WENG']
FEATURE_DIC = {'WENG_fs': SAMPLING_RATE}
RANDOM_SEED = 42

DATA_DIR = 'data/oymotion'

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, default='trained_models', help='Output directory for model')
parser.add_argument('--no-gru', action='store_true', help='Disable temporal GRU (encoder only)')
args = parser.parse_args()

global_filter = libemg.filtering.Filter(SAMPLING_RATE)
global_filter.install_common_filters()

print(f"Sampling rate: {SAMPLING_RATE} Hz | Window size: {WINDOW_SIZE} | Window inc: {WINDOW_INC} | GRU: {not args.no_gru}")

SET_NAMES = ['train', 'day1', 'limbpos']

def load_sgt_data():
    """Load SGT data (train + test sets) for evaluation during training.

    Loads per class/rep file to track file boundaries, preventing
    horizons from crossing class/rep boundaries during evaluation.
    """
    classes_list = ['0', '1', '2', '3', '4']
    reps_list = ['0', '1', '2', '3']

    datahandler = libemg.data_handler.OfflineDataHandler()
    filters = [
        libemg.data_handler.RegexFilter('C_', '_R', classes_list, 'classes'),
        libemg.data_handler.RegexFilter('R_', '_emg.csv', reps_list, 'reps'),
        libemg.data_handler.RegexFilter('/', '/', SET_NAMES, 'sets')
    ]
    sgt_dir = os.path.join(DATA_DIR, 'SGT')
    datahandler.get_data(sgt_dir, filters)
    global_filter.filter(datahandler)

    results = {}
    for i, name in enumerate(SET_NAMES):
        set_dh = datahandler.isolate_data('sets', [i])
        all_windows_list = []
        all_labels_list = []
        file_boundaries = []
        current_idx = 0

        for ci, c in enumerate(classes_list):
            class_dh = set_dh.isolate_data('classes', [ci])
            for ri, r in enumerate(reps_list):
                try:
                    rep_dh = class_dh.isolate_data('reps', [ri])
                    windows, meta = rep_dh.parse_windows(WINDOW_SIZE, WINDOW_INC)
                    if len(windows) > 0:
                        file_boundaries.append(current_idx)
                        current_idx += len(windows)
                        all_windows_list.append(windows)
                        all_labels_list.append(meta['classes'])
                except:
                    continue

        if all_windows_list:
            all_windows = np.vstack(all_windows_list)
            all_labels = np.concatenate(all_labels_list)
            results[name] = (all_windows, all_labels, file_boundaries)

    return results


def load_emg_file(filepath):
    """Load EMG data from a single file."""
    try:
        data = np.loadtxt(filepath, delimiter=' ')[:, -8:]  # Last 8 EMG channels
        return global_filter.filter(data)
    except Exception as e:
        print(f"Could not load file {filepath}: {e}")
        return None

def load_emg_data(file_list, window_size, window_inc):
    """Load EMG data from files and window it.

    Returns:
        windows: Stacked windows array (n_windows, n_channels, window_size)
        file_boundaries: List of start indices for each file's windows
    """
    all_windows = []
    file_boundaries = []
    current_idx = 0

    for filepath in file_list:
        data = load_emg_file(filepath)
        if data is not None:
            windows = libemg.utils.get_windows(data, window_size, window_inc)
            file_boundaries.append(current_idx)
            current_idx += len(windows)
            all_windows.append(windows)

    if all_windows:
        return np.vstack(all_windows), file_boundaries
    return None, []

# Discover EMG files
all_files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, '**', '*.csv'), recursive=True)
                    if '/SGT/' not in f])

print(f"Found {len(all_files)} files in {DATA_DIR}")

# 90/10 train/val split
n_val = max(1, len(all_files) // 10)
train_files = all_files[:-n_val]
val_files = all_files[-n_val:]

print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")
for f in all_files:
    split = "val" if f in val_files else "train"
    print(f"  [{split}] {f}")

# Load and window data (with file boundaries for temporal horizons)
train_windows, train_file_boundaries = load_emg_data(train_files, WINDOW_SIZE, WINDOW_INC)
val_windows, val_file_boundaries = load_emg_data(val_files, WINDOW_SIZE, WINDOW_INC)

# Extract features once to determine feature dimension
fe = libemg.feature_extractor.FeatureExtractor()
sample_features = fe.extract_features(FEATURE_LIST, train_windows[:1], feature_dic=FEATURE_DIC, array=True)
n_features = sample_features.shape[1]

print(f"Train windows shape: {train_windows.shape}")
print(f"Val windows shape: {val_windows.shape}")
print(f"Feature dimension: {n_features}")

# Load SGT data for evaluation during training
print("Loading SGT data for evaluation...")
sgt_data = load_sgt_data()
for name, (w, l, fb) in sgt_data.items():
    print(f"  SGT {name}: {w.shape}, {len(np.unique(l))} classes, {len(fb)} files")

cal_windows, cal_labels, cal_file_boundaries = sgt_data.pop('train')
test_sets = sgt_data  # remaining sets: day1, limbpos (each has file_boundaries)

os.makedirs(args.output_dir, exist_ok=True)

backbone = Backbone(n_features=n_features, use_gru=not args.no_gru)
backbone.pretrain(
    train_windows, val_windows=val_windows,
    feature_list=FEATURE_LIST,
    feature_dic=FEATURE_DIC,
    checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
    checkpoint_interval=1,
    verbose=True,
    test_sets=test_sets,
    train_labels=(cal_windows, cal_labels, cal_file_boundaries),
    batch_size=256,
    num_epochs=100,
    save_dir=args.output_dir,
    file_boundaries=train_file_boundaries,
    val_file_boundaries=val_file_boundaries,
)

print(f"Best model saved to {os.path.join(args.output_dir, 'backbone.pt')}")