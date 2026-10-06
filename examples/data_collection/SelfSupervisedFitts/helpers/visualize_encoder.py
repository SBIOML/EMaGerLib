import libemg
import torch
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from Backbone import Backbone

FEATURE_LIST = ['WENG']
WINDOW_SIZE = 200
WINDOW_INC = 50
CLASS_NAMES = ['Close', 'Open', 'Rest', 'Extension', 'Flexion']


def load_emg_file(filepath):
    """Load EMG data from a single file."""
    data = np.loadtxt(filepath, delimiter=' ')[:, 1:]  # Exclude timestamp column
    return data


def load_emg_data(file_list, window_size, window_inc):
    """Load EMG data from files and track file labels for each window."""
    all_windows = []
    all_file_labels = []
    file_names = []

    for file_idx, filepath in enumerate(file_list):
        data = load_emg_file(filepath)
        windows = libemg.utils.get_windows(data, window_size, window_inc)
        all_windows.append(windows)
        all_file_labels.append(np.full(len(windows), file_idx, dtype=np.int64))
        file_names.append(filepath)

    return np.vstack(all_windows), np.concatenate(all_file_labels), file_names


def get_encoder_embeddings(backbone, features):
    """Extract encoder embeddings from features."""
    features_scaled = backbone.scaler.transform(features)
    return backbone.embed(features_scaled)


def load_sgt_data(subject, window_size, window_inc):
    """Load SGT labelled data for a subject."""
    datahandler = libemg.data_handler.OfflineDataHandler()
    filters = [
        libemg.data_handler.RegexFilter('C_', '_R', ['0', '1', '2', '3', '4'], 'classes'),
        libemg.data_handler.RegexFilter('R_', '_emg.csv', ['0', '1', '2', '3', '4'], 'reps'),
    ]
    datahandler.get_data(f'data/SGT/subject{subject}/', filters)
    windows, meta = datahandler.parse_windows(window_size, window_inc)
    return windows, meta['classes']


def reduce_dimensionality(embeddings, method='pca'):
    """Reduce embeddings to 2D using PCA or TSNE."""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    else:
        reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_embeddings(embeddings_2d, file_labels, file_names, method='pca',
                    sgt_embeddings_2d=None, sgt_labels=None):
    """Plot 2D embeddings colored by file, with optional SGT labelled data."""
    plt.figure(figsize=(12, 8))

    # Plot unlabelled file data
    unique_labels = np.unique(file_labels)
    for label in unique_labels:
        mask = file_labels == label
        name = file_names[label].split('/')[-1]  # Just the filename
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=name, alpha=0.5, marker='o')

    # Plot SGT labelled data if provided
    if sgt_embeddings_2d is not None and sgt_labels is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
        for i, class_name in enumerate(CLASS_NAMES):
            mask = sgt_labels == i
            if np.any(mask):
                plt.scatter(sgt_embeddings_2d[mask, 0], sgt_embeddings_2d[mask, 1],
                           c=[colors[i]], label=f'SGT: {class_name}', alpha=0.8, marker='x', s=5)

    method_name = method.upper()
    plt.xlabel(f'{method_name} 1')
    plt.ylabel(f'{method_name} 2')
    plt.title(f'Encoder Embeddings ({method_name})')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def visualize_files(file_list, method='pca', subject=None):
    """Load files, extract embeddings, and visualize."""
    print(f"Loading {len(file_list)} file(s)...")
    windows, file_labels, file_names = load_emg_data(file_list, WINDOW_SIZE, WINDOW_INC)
    print(f"Total windows: {windows.shape[0]}")

    # Extract features
    fe = libemg.feature_extractor.FeatureExtractor()
    features = fe.extract_features(FEATURE_LIST, windows, array=True)
    print(f"Features shape: {features.shape}")

    # Load pretrained backbone
    backbone = Backbone(n_features=features.shape[1])
    backbone.load_state_dict(
        torch.load('trained_models/no_adversarial/backbone.pt', map_location="cpu"), strict=False,
    )
    backbone.eval()

    with open('trained_models/no_adversarial/backbone_scaler.pkl', 'rb') as f:
        backbone.scaler = pickle.load(f)

    # Get encoder embeddings
    embeddings = get_encoder_embeddings(backbone, features)
    print(f"Embeddings shape: {embeddings.shape}")

    # Load SGT data if subject is specified
    sgt_embeddings = None
    sgt_labels = None
    if subject is not None:
        print(f"Loading SGT data for subject {subject}...")
        sgt_windows, sgt_labels = load_sgt_data(subject, WINDOW_SIZE, WINDOW_INC)
        sgt_features = fe.extract_features(FEATURE_LIST, sgt_windows, array=True)
        sgt_embeddings = get_encoder_embeddings(backbone, sgt_features)
        print(f"SGT windows: {sgt_windows.shape[0]}, SGT embeddings: {sgt_embeddings.shape}")

    # Reduce dimensionality (jointly if SGT data exists)
    print(f"Reducing dimensionality with {method.upper()}...")
    if sgt_embeddings is not None:
        all_embeddings = np.vstack([embeddings, sgt_embeddings])
        all_2d = reduce_dimensionality(all_embeddings, method)
        embeddings_2d = all_2d[:len(embeddings)]
        sgt_embeddings_2d = all_2d[len(embeddings):]
    else:
        embeddings_2d = reduce_dimensionality(embeddings, method)
        sgt_embeddings_2d = None

    plot_embeddings(embeddings_2d, file_labels, file_names, method, sgt_embeddings_2d, sgt_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize encoder embeddings from EMG files')
    parser.add_argument('files', nargs='*', default=['data/subject1/emg4.csv'],
                        help='EMG file(s) to visualize (default: data/subject1/emg4.csv)')
    parser.add_argument('--method', choices=['pca', 'tsne'], default='tsne',
                        help='Dimensionality reduction method (default: tsne)')
    parser.add_argument('--subject', type=int, default=1,
                        help='Subject number to load SGT labelled data (e.g., 1 for subject1)')

    args = parser.parse_args()

    visualize_files(args.files, args.method, args.subject)