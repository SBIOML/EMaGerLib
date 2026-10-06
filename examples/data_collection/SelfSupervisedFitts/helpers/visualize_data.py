import numpy as np
import matplotlib.pyplot as plt
import libemg
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Backbone import Backbone
from sklearn.decomposition import PCA

SAMPLING_RATE = 1000
WINDOW_SIZE = 200
WINDOW_INC = 50
FEATURE_LIST = ['WENG']
FEATURE_DIC = {'WENG_fs': SAMPLING_RATE}
CLASS_NAMES = ['Close', 'Open', 'Rest', 'Extension', 'Flexion']
SET_NAMES = ['train']
MODEL_PATH = 'helpers/backbone.pt'
DATA_DIR = 'data/oymotion/Felix'

# Initialize filter
global_filter = libemg.filtering.Filter(SAMPLING_RATE)
global_filter.install_common_filters()

# Load SGT data with labels, per-file to respect file boundaries
datahandler = libemg.data_handler.OfflineDataHandler()
classes_list = ['0', '1', '2', '3', '4']
reps_list = ['0', '1', '2', '3']
filters = [
    libemg.data_handler.RegexFilter('C_', '_R', classes_list, 'classes'),
    libemg.data_handler.RegexFilter('R_', '_emg.csv', reps_list, 'reps'),
    libemg.data_handler.RegexFilter('/', '/', SET_NAMES, 'sets')
]
datahandler.get_data(DATA_DIR, filters)
global_filter.filter(datahandler)

fe = libemg.feature_extractor.FeatureExtractor()

sgt_data = {}
for i, name in enumerate(SET_NAMES):
    set_dh = datahandler.isolate_data('sets', [i])
    all_windows_list = []
    all_labels_list = []
    file_boundaries = []
    current_idx = 0

    # Iterate per class/rep to get per-file windows
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
        sgt_data[name] = (all_windows, all_labels, file_boundaries)
        print(f"  {name}: {all_windows.shape}, {len(np.unique(all_labels))} classes, {len(file_boundaries)} files")

# Determine feature dimension
sample_feats = fe.extract_features(FEATURE_LIST, sgt_data['train'][0][:1], feature_dic=FEATURE_DIC, array=True)
n_features = sample_feats.shape[1]

# Load backbone
backbone = Backbone(n_features=n_features)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
backbone.load_state_dict(state_dict, strict=False)
backbone.eval()
backbone.feature_list = FEATURE_LIST
backbone.feature_dic = FEATURE_DIC
print(f"Loaded model from {MODEL_PATH}")

# Get embeddings for each set, building horizons per-file to avoid cross-file contamination
embeddings = {}
raw_features = {}
for name, (windows, labels, boundaries) in sgt_data.items():
    feats = fe.extract_features(FEATURE_LIST, windows, feature_dic=FEATURE_DIC, array=True)
    raw_features[name] = feats

    # Build horizons per file and concatenate
    all_emb = []
    for fi in range(len(boundaries)):
        start = boundaries[fi]
        end = boundaries[fi + 1] if fi + 1 < len(boundaries) else len(feats)
        file_feats = feats[start:end]
        all_emb.append(backbone.embed(file_feats))
    embeddings[name] = np.vstack(all_emb)

# --- Plot: Raw features vs Encoder embeddings (PCA) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

class_colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
set_markers = {'train': 'o', 'day1': 's', 'limbpos': 'D'}
set_sizes = {'train': 50, 'day1': 30, 'limbpos': 30}

for ax, (data_dict, title) in zip(axes, [(raw_features, 'Raw WENG Features (PCA)'),
                                          (embeddings, 'GRU Encoder Embeddings (PCA)')]):
    # Joint PCA across all sets
    all_data = np.vstack([data_dict[name] for name in SET_NAMES])
    pca = PCA(n_components=2, random_state=42)
    pca_all = pca.fit_transform(all_data)

    offset = 0
    for name in SET_NAMES:
        n = len(data_dict[name])
        pca_set = pca_all[offset:offset + n]
        labels = sgt_data[name][1]
        offset += n

        for c in range(len(CLASS_NAMES)):
            mask = labels == c
            ax.scatter(pca_set[mask, 0], pca_set[mask, 1],
                       c=[class_colors[c]], marker=set_markers[name],
                       s=set_sizes[name], alpha=0.6,
                       label=f'{CLASS_NAMES[c]} ({name})' if name == 'train' else None)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(title)

# Build legend
from matplotlib.lines import Line2D
legend_elements = []
for i, cn in enumerate(CLASS_NAMES):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
                                  markersize=8, label=cn))
legend_elements.append(Line2D([0], [0], color='w', label=''))
for name, marker in set_markers.items():
    legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                                  markersize=8, label=name))
fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5))
plt.suptitle('Raw Features vs GRU Encoder Embeddings', fontsize=14)
plt.tight_layout(rect=[0, 0, 0.90, 0.96])
plt.show()
