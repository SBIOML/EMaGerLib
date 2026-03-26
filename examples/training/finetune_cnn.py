from pathlib import Path
import logging

from emagerlib import ROOT_EMAGERLIB
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested


# ============================================================
# Default configuration path
# ============================================================
DEFAULT_CONFIG = ROOT_EMAGERLIB / "examples" / "training" / "config_train-model.py"

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Train / fine-tune CNN model on EMG data",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="train_cnn")
    save_config_if_requested(args, cfg, script_name="train_cnn")
    return args, cfg


# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_data(dataset_folder, cfg):
    from libemg.data_handler import OfflineDataHandler, RegexFilter
    from libemg.filtering import Filter

    classes_values = [str(num) for num in range(cfg.NUM_CLASSES)]
    reps_values = [str(num) for num in range(cfg.NUM_REPS)]

    regex_filters = [
        RegexFilter(left_bound="C_", right_bound="_", values=classes_values, description="classes"),
        RegexFilter(left_bound="R_", right_bound="_emg.csv", values=reps_values, description="reps"),
    ]

    odh = OfflineDataHandler()
    odh.get_data(folder_location=dataset_folder, regex_filters=regex_filters)

    filt = Filter(cfg.SAMPLING)

    notch_filter_dictionary = {"name": "notch", "cutoff": 60, "bandwidth": 3}
    filt.install_filters(notch_filter_dictionary)

    bandpass_filter_dictionary = {"name": "bandpass", "cutoff": [20, 450], "order": 4}
    filt.install_filters(bandpass_filter_dictionary)

    filt.filter(odh)
    return odh


# ============================================================
# MODEL HELPERS
# ============================================================
def load_pretrained_weights(model, model_path, device="cpu"):
    """
    Load a saved state_dict into the model.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")

    import torch

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded pretrained model from: {model_path}")


def freeze_first_n_parameters(model, n_to_freeze):
    """
    Freeze the first n parameter tensors in model.named_parameters().
    """
    logger.info("Parameter list before freezing:")
    for idx, (name, param) in enumerate(model.named_parameters()):
        logger.info(f"[{idx}] {name} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}")

    frozen_names = []
    unfrozen_names = []

    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx < n_to_freeze:
            param.requires_grad = False
            frozen_names.append(name)
        else:
            param.requires_grad = True
            unfrozen_names.append(name)

    logger.info(f"Froze first {n_to_freeze} parameter tensors:")
    for name in frozen_names:
        logger.info(f"  FROZEN   : {name}")

    logger.info("Remaining trainable parameter tensors:")
    for name in unfrozen_names:
        logger.info(f"  TRAINABLE: {name}")


def print_trainable_summary(model):
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n

    logger.info(f"Total parameters     : {total_params}")
    logger.info(f"Trainable parameters : {trainable_params}")
    logger.info(f"Frozen parameters    : {total_params - trainable_params}")


# ============================================================
# MAIN
# ============================================================
def main(argv=None):
    import datetime
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from libemg.feature_extractor import FeatureExtractor
    import emagerlib.models.models as etm

    _, cfg = setup_runtime(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data = prepare_data(cfg.DATASETS_PATH, cfg)

    # Split data into training and testing
    train_data_obj = data.isolate_data("reps", cfg.TRAIN_REPS)
    test_data_obj = data.isolate_data("reps", cfg.TEST_REPS)

    # Extract windows
    train_windows, train_meta = train_data_obj.parse_windows(cfg.WINDOW_SIZE, cfg.WINDOW_INCREMENT)
    test_windows, test_meta = test_data_obj.parse_windows(cfg.WINDOW_SIZE, cfg.WINDOW_INCREMENT)

    logger.info(f"Training metadata: {train_meta}, Testing metadata: {test_meta}")
    logger.info(f"Training windows: {train_windows.shape}, Testing windows: {test_windows.shape}")

    # Features extraction
    fe = FeatureExtractor()
    train_features = fe.getMAVfeat(train_windows)
    train_labels = train_meta["classes"]
    test_features = fe.getMAVfeat(test_windows)
    test_labels = test_meta["classes"]

    # Optional PCA visualization
    features_data = {"key": train_features}
    fe.visualize_feature_space(features_data, "PCA", classes=train_labels, render=False)

    logger.info(f"Training features: {train_features.shape}, Testing features: {test_features.shape}")

    train_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features.astype(np.float32)),
            torch.from_numpy(train_labels).long()
        ),
        batch_size=64,
        shuffle=True,
    )

    test_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(test_features.astype(np.float32)),
            torch.from_numpy(test_labels).long()
        ),
        batch_size=256,
        shuffle=False,
    )

    # Create model
    classifier = etm.EmagerCNN((4, 16), cfg.NUM_CLASSES, -1)

    # Load pretrained weights if requested
    if cfg.USE_PRETRAINED:
        load_pretrained_weights(classifier, cfg.PRETRAINED_MODEL_PATH, device=device)

        # Freeze first few layers/parameter tensors
        freeze_first_n_parameters(classifier, cfg.FREEZE_FIRST_N_PARAMS)
        print_trainable_summary(classifier)
    else:
        logger.info("Training from scratch (no pretrained weights loaded).")

    # Move model after loading
    classifier = classifier.to(device)

    # Fine-tune / train
    start_time = time.time()
    res = classifier.fit(train_dl, test_dl, max_epochs=cfg.EPOCH)
    elapsed = time.time() - start_time

    acc = int(res[0]["test_acc"] * 1000)
    logger.info(f"Resultat: {res} accuracy : {acc}/1000")
    logger.info(f"Training/fine-tuning time: {elapsed:.2f} s")

    current_time = datetime.datetime.now().strftime("%y-%m-%d_%Hh%M")
    logger.info(f"Current time: {current_time}")

    # Save fine-tuned model
    model_path = Path(cfg.SAVE_PATH) / f"libemg_torch_cnn_finetuned_{cfg.SESSION}_{acc}_{current_time}.pth"
    torch.save(classifier.state_dict(), model_path)
    logger.info(f"Model saved at {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())