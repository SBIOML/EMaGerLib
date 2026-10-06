import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import libemg
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from libemg.feature_extractor import FeatureExtractor
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss


# ============== Raw EMG Augmentations for SimCLR ==============
# All augmentations expect input shape: (n_channels, window_size)
class GaussianNoise:
    """Add Gaussian noise to raw EMG signals."""
    def __init__(self, snr_db=15):
        self.snr_db = snr_db

    def __call__(self, x):
        """
        Args:
            x: Raw EMG window, shape (n_channels, window_size), numpy array
        """
        signal_power = np.mean(x ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(*x.shape) * np.sqrt(noise_power)
        return x + noise


class MagnitudeScaling:
    """Scale the magnitude of raw EMG signals by a random factor."""
    def __init__(self, scale_range=(0.5, 2)):
        self.scale_range = scale_range

    def __call__(self, x):
        """
        Args:
            x: Raw EMG window, shape (n_channels, window_size), numpy array
        """
        scale = np.random.uniform(*self.scale_range)
        return x * scale


class PerChannelMagnitudeScaling:
    """Scale each channel independently to simulate varying skin-electrode impedance."""
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, x):
        n_channels = x.shape[0]
        scales = np.random.uniform(*self.scale_range, size=(n_channels, 1))
        return x * scales


class ChannelRotation:
    """Rotate/permute EMG channels to simulate electrode placement variation.

    Useful when electrodes are arranged in a ring around a limb.
    """
    def __init__(self, rotation_range=1):
        self.rotation_range = rotation_range

    def __call__(self, x):
        """
        Args:
            x: Raw EMG window, shape (n_channels, window_size), numpy array
        """
        shift = random.randint(-self.rotation_range, self.rotation_range)
        if shift == 0:
            return x
        return np.roll(x, shift=shift, axis=0)

class ChannelDropout:
    """Randomly zero out entire channels to improve robustness."""
    def __init__(self, dropout_prob=0.1):
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        """
        Args:
            x: Raw EMG window, shape (n_channels, window_size), numpy array
        """
        n_channels = x.shape[0]
        mask = (np.random.rand(n_channels) > self.dropout_prob).astype(np.float32).reshape(-1, 1)
        return x * mask

class TemporalShift:
    """Randomly shifts the signal in time, wrapping around."""
    def __init__(self, max_shift=0.2):
        self.max_shift = max_shift

    def __call__(self, x):
        shift = int(x.shape[1] * random.uniform(-self.max_shift, self.max_shift))
        return np.roll(x, shift, axis=1)

class EMGAugmentationPipeline:
    """Compose multiple augmentations for raw EMG data."""
    def __init__(self, augmentations, p=0.25):
        self.augmentations = augmentations
        self.p = p  # Probability of applying each augmentation

    def __call__(self, x):
        """
        Args:
            x: Raw EMG window, shape (n_channels, window_size), numpy array
        """
        x = x.copy()  # Don't modify original
        for aug in self.augmentations:
            if random.random() < self.p:
                x = aug(x)
        return x

    def augment_horizon(self, windows):
        """Apply the same augmentation choices to all windows in a horizon.

        Args:
            windows: Array of shape (horizon, n_channels, window_size)

        Returns:
            Augmented windows with same shape, same augmentations applied to each window
        """
        windows = windows.copy()
        # Pre-decide which augmentations to apply
        apply_mask = [random.random() < self.p for _ in self.augmentations]

        # Save and restore random state so each augmentation uses the same params per window
        for i, (aug, should_apply) in enumerate(zip(self.augmentations, apply_mask)):
            if should_apply:
                # Use the same random seed for all windows in the horizon
                seed = random.randint(0, 2**31)
                for j in range(len(windows)):
                    np.random.seed(seed)
                    random.seed(seed)
                    windows[j] = aug(windows[j])

        # Reset random state to avoid affecting downstream randomness
        np.random.seed(None)
        random.seed()
        return windows


def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SimCLRDataset(Dataset):
    """Dataset that returns two augmented views of horizons for temporal SimCLR.

    Applies per-window augmentations (noise, scaling, etc.) so the GRU must learn temporal structure.
    """
    def __init__(self, windows, augmentation_pipeline, feature_list=['WENG'],
                 file_boundaries=None, horizon=15, stride=1):
        """
        Args:
            windows: Raw EMG windows, shape (n_windows, n_channels, window_size), numpy array
            augmentation_pipeline: Per-window augmentation pipeline
            feature_list: List of features to extract using libemg FeatureExtractor
            file_boundaries: List of start indices for each file's windows (to avoid cross-file horizons)
            horizon: Number of consecutive windows per horizon
            stride: Step size between horizon start indices (default 1, use >1 to reduce dataset size)
        """
        self.data = windows
        self.augmentation = augmentation_pipeline
        self.feature_list = feature_list
        self.horizon = horizon

        # Build valid horizon start indices (sequences that don't cross file boundaries)
        if file_boundaries is not None:
            self.valid_indices = []
            for i in range(len(file_boundaries)):
                start = file_boundaries[i]
                end = file_boundaries[i + 1] if i + 1 < len(file_boundaries) else len(windows)
                for j in range(start, end - horizon + 1, stride):
                    self.valid_indices.append(j)
            self.valid_indices = np.array(self.valid_indices)
        else:
            self.valid_indices = np.arange(0, len(windows) - horizon + 1, stride)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        horizon_windows = self.data[start:start + self.horizon]  # (horizon, n_channels, window_size)

        # Per-window augmentations (same params within each horizon, different between views)
        aug1 = self.augmentation.augment_horizon(horizon_windows)
        aug2 = self.augmentation.augment_horizon(horizon_windows)

        return aug1, aug2

    def __len__(self):
        return len(self.valid_indices)


def create_simclr_collate_fn(feature_list, feature_dic=None):
    """Create a collate function that batches feature extraction for horizons."""
    feature_extractor = FeatureExtractor()

    def collate_fn(batch):
        aug1_list, aug2_list = zip(*batch)

        # Stack horizons: (batch, horizon, channels, window_size)
        aug1_batch = np.stack(aug1_list, axis=0)
        aug2_batch = np.stack(aug2_list, axis=0)

        batch_size = aug1_batch.shape[0]
        h = aug1_batch.shape[1]

        # Flatten batch and horizon for feature extraction: (batch*horizon, channels, window_size)
        aug1_flat = aug1_batch.reshape(-1, *aug1_batch.shape[2:])
        aug2_flat = aug2_batch.reshape(-1, *aug2_batch.shape[2:])

        # Extract features in batch
        features1 = feature_extractor.extract_features(feature_list, aug1_flat, feature_dic=feature_dic, array=True)
        features2 = feature_extractor.extract_features(feature_list, aug2_flat, feature_dic=feature_dic, array=True)

        n_features = features1.shape[1]

        if h == 1:
            # No GRU: return (batch, n_features)
            view1 = torch.tensor(features1, dtype=torch.float32)
            view2 = torch.tensor(features2, dtype=torch.float32)
        else:
            # GRU: return (batch, horizon, n_features)
            view1 = torch.tensor(features1.reshape(batch_size, h, n_features), dtype=torch.float32)
            view2 = torch.tensor(features2.reshape(batch_size, h, n_features), dtype=torch.float32)

        return view1, view2

    return collate_fn

class Backbone(nn.Module):
    def __init__(self, n_features, layers=[128, 128, 128], dropout_p=0.1, projection_dim=128, horizon=15, use_gru=True):
        super().__init__()
        fix_random_seed(0)

        # Per-window augmentations: invariance to sensor confounds
        self.augmentations = EMGAugmentationPipeline([
            GaussianNoise(snr_db=15),
            PerChannelMagnitudeScaling(scale_range=(0.5, 2.0)),
            ChannelDropout(dropout_prob=0.1),
            ChannelRotation(rotation_range=1),
            TemporalShift(max_shift=0.4),
        ], p=0.5)

        self.n_features = n_features
        self.layers = layers
        self.hidden_size = layers[-1]
        self.horizon = horizon
        self.use_gru = use_gru

        # MLP encoder: per-window feature transform
        self.encoder = nn.Sequential(
            nn.Linear(n_features, layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.LayerNorm(layers[-1])
        )

        # GRU: temporal dynamics over MLP-encoded windows
        if use_gru:
            self.gru = nn.GRU(input_size=layers[-1], hidden_size=layers[-1], num_layers=1, batch_first=True)

        # Projection head for SimCLR (maps to embedding space for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Linear(layers[-1], layers[-1]),
            nn.ReLU(),
            nn.Linear(layers[-1], projection_dim)
        )

        self.log = {"pretrain_loss": []}

    def forward(self, x):
        """Run through MLP encoder, optionally followed by GRU.

        Args:
            x: (batch, n_features) if no GRU, (batch, horizon, n_features) if GRU

        Returns:
            (batch, hidden_size)
        """
        if self.use_gru:
            batch_size, horizon, n_feat = x.shape
            x_flat = x.reshape(batch_size * horizon, n_feat)
            encoded = self.encoder(x_flat)
            encoded = encoded.reshape(batch_size, horizon, -1)
            _, h_n = self.gru(encoded)  # h_n: (1, batch, hidden_size)
            return h_n.squeeze(0)
        else:
            return self.encoder(x)

    def project(self, x):
        """Get projection embeddings for contrastive learning."""
        features = self.forward(x)
        return self.projection_head(features)

    def pretrain(self, train_windows, val_windows=None, feature_list=['WENG'], feature_dic=None, learning_rate=1e-3, num_epochs=50,
                 batch_size=512, temperature=0.1, checkpoint_dir=None, checkpoint_interval=10, verbose=False,
                 test_sets=None, train_labels=None, save_dir=None,
                 file_boundaries=None, val_file_boundaries=None, train_stride=4):
        """
        Pretrain the encoder using SimCLR on unlabeled data.

        Args:
            train_windows: Raw EMG windows, shape (n_windows, n_channels, window_size), numpy array
            val_windows: Validation EMG windows, optional
            feature_list: List of features to extract using libemg FeatureExtractor
            learning_rate: Learning rate for optimizer
            num_epochs: Number of pretraining epochs
            batch_size: Batch size for training
            temperature: Temperature for NTXent loss
            checkpoint_dir: Directory to save checkpoints (None to disable)
            checkpoint_interval: Save checkpoint every N epochs
            verbose: Print progress
            test_sets: Optional dict of test sets for per-epoch evaluation.
                       Format: {'name': (windows, labels, file_boundaries)} where windows are raw EMG windows,
                       labels are class labels, and file_boundaries are start indices per file.
            train_labels: Tuple of (calibration_windows, calibration_labels, file_boundaries) used to calibrate
                          the classifier before evaluating test sets each epoch
        """
        import os
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.feature_list = feature_list
        self.feature_dic = feature_dic

        dataset_horizon = self.horizon if self.use_gru else 1
        train_dataset = SimCLRDataset(train_windows, self.augmentations,
                                       feature_list=self.feature_list,
                                       file_boundaries=file_boundaries, horizon=dataset_horizon,
                                       stride=train_stride)
        train_collate = create_simclr_collate_fn(self.feature_list, feature_dic=self.feature_dic)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  collate_fn=train_collate, num_workers=4, persistent_workers=True)

        val_loader = None
        if val_windows is not None:
            val_dataset = SimCLRDataset(val_windows, self.augmentations,
                                        feature_list=self.feature_list,
                                        file_boundaries=val_file_boundaries, horizon=dataset_horizon)
            val_collate = create_simclr_collate_fn(self.feature_list, feature_dic=self.feature_dic)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                    collate_fn=val_collate, num_workers=2, persistent_workers=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        params = list(self.encoder.parameters()) + list(self.projection_head.parameters())
        if self.use_gru:
            params += list(self.gru.parameters())
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

        ssl_loss = SelfSupervisedLoss(NTXentLoss(temperature=temperature))

        pretrain_losses = []
        self.log["pretrain_val_loss"] = []
        self.log["test_set_accuracy"] = {name: [] for name in test_sets} if test_sets else {}
        best_avg_acc = -1.0

        for epoch in range(num_epochs):
            self.train()
            batch_losses = []

            for batch in train_loader:
                view1, view2 = batch
                view1, view2 = view1.to(device), view2.to(device)

                optimizer.zero_grad()

                features1 = self.forward(view1)
                features2 = self.forward(view2)
                z1 = self.projection_head(features1)
                z2 = self.projection_head(features2)

                loss = ssl_loss(z1, z2)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            avg_train_loss = np.mean(batch_losses)
            pretrain_losses.append(avg_train_loss)
            scheduler.step()

            self.log["pretrain_loss"].append((epoch, avg_train_loss))

            # Validation
            avg_val_loss = None
            if val_loader is not None:
                self.eval()
                val_batch_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        view1, view2 = batch
                        view1, view2 = view1.to(device), view2.to(device)
                        z1 = self.project(view1)
                        z2 = self.project(view2)
                        val_loss = ssl_loss(z1, z2)
                        val_batch_losses.append(val_loss.item())
                avg_val_loss = np.mean(val_batch_losses)
                self.log["pretrain_val_loss"].append((epoch, avg_val_loss))

            # Per-epoch downstream evaluation on test sets
            epoch_accs = {}
            if test_sets is not None and train_labels is not None:
                cal_windows, cal_labels, cal_boundaries = train_labels
                self.eval()
                self.to('cpu')
                self.calibrate(cal_windows, cal_labels, file_boundaries=cal_boundaries)
                fe_eval = FeatureExtractor()
                for name, (test_windows, test_labels, test_boundaries) in test_sets.items():
                    test_features = fe_eval.extract_features(self.feature_list, test_windows, feature_dic=self.feature_dic, array=True)
                    preds = self.predict(test_features, file_boundaries=test_boundaries)
                    acc = libemg.offline_metrics.OfflineMetrics().get_CA(test_labels, preds)
                    epoch_accs[name] = acc
                    self.log["test_set_accuracy"][name].append((epoch, acc))
                if save_dir is not None and epoch_accs:
                    avg_acc = np.mean(list(epoch_accs.values()))
                    if avg_acc > best_avg_acc:
                        best_avg_acc = avg_acc
                        best_model_path = os.path.join(save_dir, 'backbone.pt')
                        torch.save(self.state_dict(), best_model_path)
                        if verbose:
                            print(f"  ** New best avg accuracy: {avg_acc:.4f} — saved to {best_model_path}")

                self.to(device)
                self.train()

            if verbose:
                val_str = f" | Val Loss {avg_val_loss:.4f}" if avg_val_loss is not None else ""
                test_str = "".join(f" | {name} {acc:.2f}" for name, acc in epoch_accs.items()) if epoch_accs else ""
                print(f"Pretrain Epoch {epoch}: SimCLR Loss {avg_train_loss:.4f}{val_str}{test_str}")

            # Save checkpoint every N epochs
            if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                if verbose:
                    print(f"Checkpoint saved to {checkpoint_path}")

        self.eval()
        self.to('cpu')

        if verbose:
            print("Pretraining complete. Use calibrate() and predict() for classification.")

    def _build_horizons(self, features):
        """Build sliding-window horizons from sequential features.

        For window i, the horizon is features[i-horizon+1 : i+1].
        Early windows (i < horizon-1) are left-padded by repeating the first window.

        Args:
            features: (n_windows, n_features) numpy array

        Returns:
            horizons: (n_windows, horizon, n_features) numpy array
        """
        n = len(features)
        h = self.horizon
        # Pad the beginning with repeats of the first window
        padded = np.concatenate([np.tile(features[0:1], (h - 1, 1)), features], axis=0)
        # Build horizons via sliding window
        horizons = np.stack([padded[i:i + h] for i in range(n)], axis=0)
        return horizons

    def _embed_per_file(self, features, file_boundaries):
        """Run embed per file to avoid cross-file horizon contamination.

        Args:
            features: (n_windows, n_features) numpy array
            file_boundaries: List of start indices for each file's windows

        Returns:
            embeddings: (n_windows, hidden_size) numpy array, L2-normalized
        """
        all_emb = []
        for fi in range(len(file_boundaries)):
            start = file_boundaries[fi]
            end = file_boundaries[fi + 1] if fi + 1 < len(file_boundaries) else len(features)
            all_emb.append(self.embed(features[start:end]))
        return np.vstack(all_emb)

    def embed(self, features):
        """Run features through the encoder (and GRU if enabled).

        Args:
            features: (n_windows, n_features) numpy array

        Returns:
            embeddings: (n_windows, hidden_size) numpy array, L2-normalized
        """
        with torch.no_grad():
            if self.use_gru:
                horizons = self._build_horizons(features)
                embeddings = self.forward(torch.tensor(horizons, dtype=torch.float32)).numpy()
            else:
                embeddings = self.forward(torch.tensor(features, dtype=torch.float32)).numpy()
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return embeddings

    def calibrate(self, calibration_features_raw, calibration_labels, n_neighbors=5, file_boundaries=None):
        """
        Calibrate a kNN classifier on encoder embeddings.

        Args:
            calibration_features_raw: Raw EMG windows for calibration
            calibration_labels: Class labels for each window
            n_neighbors: Number of neighbors for kNN (default=5)
            file_boundaries: List of start indices for each file's windows (to avoid cross-file horizons)
        """
        self.eval()
        fe = libemg.feature_extractor.FeatureExtractor()
        f = fe.extract_features(self.feature_list, calibration_features_raw, feature_dic=self.feature_dic, array=True)

        if file_boundaries is not None:
            embeddings = self._embed_per_file(f, file_boundaries)
        else:
            embeddings = self.embed(f)

        self.classes = np.unique(calibration_labels)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        self.knn_classifier.fit(embeddings, calibration_labels)

    def predict(self, features, file_boundaries=None):
        """
        Classify using kNN on encoder embeddings.

        Args:
            features: EMG features to classify (numpy array)
            file_boundaries: List of start indices for each file's windows (to avoid cross-file horizons)
        """
        if not hasattr(self, 'knn_classifier'):
            raise RuntimeError("Call calibrate() first")

        self.eval()
        if file_boundaries is not None:
            embeddings = self._embed_per_file(features, file_boundaries)
        else:
            embeddings = self.embed(features)

        return self.knn_classifier.predict(embeddings)

    def predict_proba(self, features):
        """
        Return class probabilities as one-hot encoded predictions.

        Args:
            features: EMG features to classify (numpy array)

        Returns:
            probabilities: Array of shape (n_samples, n_classes) with one-hot encoded predictions
        """
        predictions = self.predict(features)

        # One-hot encode predictions
        n_samples = len(predictions)
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes == pred)[0][0]
            probabilities[i, class_idx] = 1.0

        return probabilities