import libemg
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Backbone import Backbone
from MLP import MLP

# Run Model Online
run = False

# Plot embedding
plot = True

FEATURE_LIST = ['WENG']
SAMPLING_RATE = 1000
WINDOW_SIZE = 200
WINDOW_INC = 50
feature_dic = {'WENG_fs': SAMPLING_RATE}

global_filter = libemg.filtering.Filter(SAMPLING_RATE)
global_filter.install_common_filters()

classes_list = ['0', '1', '2', '3', '4']
reps_list = ['0', '1', '2', '3']


def load_set_per_file(datahandler, set_idx):
    """Load a set's data, windowed per class/rep file to track boundaries.

    Returns:
        windows: (n_windows, n_channels, window_size)
        labels: (n_windows,) class labels
        file_boundaries: list of start indices for each file
    """
    set_dh = datahandler.isolate_data('sets', [set_idx])
    all_windows = []
    all_labels = []
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
                    all_windows.append(windows)
                    all_labels.append(meta['classes'])
            except:
                continue

    return np.vstack(all_windows), np.concatenate(all_labels), file_boundaries


def embed_per_file(backbone, feats, file_boundaries):
    """Run embed per file to avoid cross-file horizon contamination."""
    all_emb = []
    for fi in range(len(file_boundaries)):
        start = file_boundaries[fi]
        end = file_boundaries[fi + 1] if fi + 1 < len(file_boundaries) else len(feats)
        all_emb.append(backbone.embed(feats[start:end]))
    return np.vstack(all_emb)


if __name__ == "__main__":
    datahandler = libemg.data_handler.OfflineDataHandler()
    filters = [
        libemg.data_handler.RegexFilter('C_', '_R', classes_list, 'classes'),
        libemg.data_handler.RegexFilter('R_', '_emg.csv', reps_list, 'reps'),
        libemg.data_handler.RegexFilter('/', '/', ['train', 'day1'], 'sets')
    ]
    datahandler.get_data('data/oymotion/Felix/', filters)
    global_filter.filter(datahandler)

    set_names = ['train', 'day1']
    metrics = libemg.offline_metrics.OfflineMetrics()
    class_names = ['Close', 'Open', 'Rest', 'Extension', 'Flexion']
    fe = libemg.feature_extractor.FeatureExtractor()

    # Load train set with per-file boundaries
    tr_windows, tr_labels, tr_boundaries = load_set_per_file(datahandler, 0)
    tr_feats = fe.extract_features(['WENG'], tr_windows, array=True, feature_dic=feature_dic)
    print(tr_feats.shape)


    def plot_pca_side_by_side(tr_raw, tr_encoder, tr_labels, test_raw_dict, test_encoder_dict):
        """Plot raw features vs encoder features side by side in PCA space."""
        from matplotlib.lines import Line2D

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        class_colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        set_markers = {'train': 'o', 'day1': 's', 'limbpos': 'D'}
        set_alphas = {'train': 0.9, 'day1': 0.6, 'limbpos': 0.6}
        set_sizes = {'train': 60, 'day1': 40, 'limbpos': 40}

        data_configs = [
            (tr_raw, test_raw_dict, 'Raw WENG Features', axes[0]),
            (tr_encoder, test_encoder_dict, f'Encoder Embeddings', axes[1])
        ]

        for tr_data, test_dict, title, ax in data_configs:
            # Combine all embeddings for joint PCA
            all_data = [tr_data]
            for set_name in test_dict:
                all_data.append(test_dict[set_name]['embeddings'])
            all_data = np.vstack(all_data)

            pca = PCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(all_data)

            # Split back
            n_train = len(tr_data)
            tr_pca = pca_result[:n_train]

            # Split test set results
            te_pca_dict = {}
            start_idx = n_train
            for set_name in test_dict:
                n_samples = len(test_dict[set_name]['embeddings'])
                te_pca_dict[set_name] = pca_result[start_idx:start_idx + n_samples]
                start_idx += n_samples

            # Plot train points
            for i, class_name in enumerate(class_names):
                tr_mask = tr_labels == i
                ax.scatter(tr_pca[tr_mask, 0], tr_pca[tr_mask, 1],
                          c=[class_colors[i]], alpha=set_alphas['train'],
                          marker=set_markers['train'], s=set_sizes['train'])

                # Plot test points for each set
                for set_name in test_dict:
                    te_labels = test_dict[set_name]['labels']
                    te_pca = te_pca_dict[set_name]
                    te_mask = te_labels == i
                    ax.scatter(te_pca[te_mask, 0], te_pca[te_mask, 1],
                              c=[class_colors[i]], alpha=set_alphas[set_name],
                              marker=set_markers[set_name], s=set_sizes[set_name])

            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(title)

        # Create shared legend
        legend_elements = []
        for i, class_name in enumerate(class_names):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
                                         markersize=10, label=class_name))
        legend_elements.append(Line2D([0], [0], color='w', label=''))
        for set_name, marker in set_markers.items():
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                                         markersize=10, label=set_name))

        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5))
        plt.suptitle('Raw Features vs Encoder Embeddings (PCA)', fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
        plt.show()


    backbone = Backbone(n_features=tr_feats.shape[1])
    model = 'trained_models/backbone.pt'
    checkpoint = torch.load(model, map_location="cpu")
    # Support both raw state_dicts (backbone.pt) and checkpoint dicts
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    backbone.load_state_dict(state_dict, strict=False)
    backbone.eval()
    backbone.feature_list = ['WENG']
    backbone.feature_dic = feature_dic

    # Get encoder embeddings for train (per-file horizons)
    tr_embeddings = embed_per_file(backbone, tr_feats, tr_boundaries)

    # Methods: encoder-based and raw-feature-based
    encoder_methods = ['KNN (enc)', 'LDA (enc)', 'MLP (enc)']
    raw_methods = ['KNN (raw)', 'LDA (raw)', 'MLP (raw)']
    all_methods = encoder_methods + raw_methods

    # Store results for all sets
    results = {method: {} for method in all_methods}

    # Collect all test embeddings for unified plotting
    test_embeddings_dict = {}
    test_raw_feats_dict = {}

    # Pre-fit raw classifiers
    from sklearn.neighbors import KNeighborsClassifier
    lda = LinearDiscriminantAnalysis()
    lda.fit(tr_feats, tr_labels)
    knn_raw = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_raw.fit(tr_feats, tr_labels)

    # Pre-fit LDA on encoder embeddings
    lda_enc = LinearDiscriminantAnalysis()
    lda_enc.fit(tr_embeddings, tr_labels)

    # Pre-fit MLP on raw features
    mlp_raw = MLP()
    mlp_raw.calibrate(tr_feats, tr_labels)

    # Pre-fit MLP on encoder embeddings
    mlp_enc = MLP()
    mlp_enc.calibrate(tr_embeddings, tr_labels)

    # Iterate over all test sets (1-4)
    for set_idx in range(1, len(set_names)):
        set_name = set_names[set_idx]
        te_windows, te_labels_set, te_boundaries = load_set_per_file(datahandler, set_idx)
        te_feats = fe.extract_features(['WENG'], te_windows, array=True, feature_dic=feature_dic)

        # Get encoder embeddings for test (per-file horizons)
        te_embeddings = embed_per_file(backbone, te_feats, te_boundaries)

        # Store for plotting
        test_embeddings_dict[set_name] = {
            'embeddings': te_embeddings,
            'labels': te_labels_set
        }
        test_raw_feats_dict[set_name] = {
            'embeddings': te_feats,
            'labels': te_labels_set
        }

        # --- Encoder-based classifiers ---
        # KNN (enc): calibrate and predict using per-file embeddings
        backbone.calibrate(tr_windows, tr_labels, file_boundaries=tr_boundaries)
        te_preds = backbone.predict(te_feats, file_boundaries=te_boundaries)
        results['KNN (enc)'][set_name] = metrics.get_CA(te_preds, te_labels_set)
        results['LDA (enc)'][set_name] = metrics.get_CA(lda_enc.predict(te_embeddings), te_labels_set)
        results['MLP (enc)'][set_name] = metrics.get_CA(mlp_enc.predict(te_embeddings), te_labels_set)

        # --- Raw feature classifiers (no encoder) ---
        results['KNN (raw)'][set_name] = metrics.get_CA(knn_raw.predict(te_feats), te_labels_set)
        results['LDA (raw)'][set_name] = metrics.get_CA(lda.predict(te_feats), te_labels_set)
        results['MLP (raw)'][set_name] = metrics.get_CA(mlp_raw.predict(te_feats), te_labels_set)

    # Plot raw vs encoder embeddings side by side
    if plot:
        plot_pca_side_by_side(tr_feats, tr_embeddings, tr_labels,
                              test_raw_feats_dict, test_embeddings_dict)

    # Summary
    print("\n" + "="*80)
    print("RESULTS (Training on 'train' set)")
    print("="*80)
    header = f"{'Method':<22}" + "".join(f"{name:<12}" for name in set_names[1:])
    print(header)
    print("-"*80)
    print("WITH ENCODER:")
    for method in encoder_methods:
        row = f"  {method:<20}" + "".join(f"{results[method][name]:<12.4f}" for name in set_names[1:])
        print(row)
    print("WITHOUT ENCODER (raw features):")
    for method in raw_methods:
        row = f"  {method:<20}" + "".join(f"{results[method][name]:<12.4f}" for name in set_names[1:])
        print(row)


    if run:
        p, smm = libemg.streamers.oymotion_streamer()
        odh = libemg.data_handler.OnlineDataHandler(smm)
        offline_model = libemg.emg_predictor.EMGClassifier(None)
        offline_model.model = backbone
        online_classifier = libemg.emg_predictor.OnlineEMGClassifier(offline_model, WINDOW_SIZE, WINDOW_INC, odh, FEATURE_LIST, std_out=True)
        online_classifier.run(block=True)
