from pathlib import Path
import logging
import datetime

from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"
logger = logging.getLogger(__name__)

# Comment / uncomment to select which variants to run.
MODELS_TO_TEST = [
    "EmagerCNNBase",
    "EmagerCNNWide",
    "EmagerCNNDeep",
    "EmagerCNNLight",
    "EmagerCNNStrided",
    "EmagerCNNCircular",
    "EmagerCNNGAP",
]


def parse_args(argv=None):
    parser = create_parser(
        description="Train and compare new EmagerCNN variants",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="train_cnn_new")
    save_config_if_requested(args, cfg, script_name="train_cnn_new")
    return args, cfg


def load_and_filter(dataset_folder, cfg):
    """Load raw EMG from disk and apply notch + bandpass filters."""
    from libemg.data_handler import OfflineDataHandler, RegexFilter
    from libemg.filtering import Filter

    classes_values = [str(i) for i in range(cfg.NUM_CLASSES)]
    reps_values    = [str(i) for i in range(cfg.NUM_REPS)]

    odh = OfflineDataHandler()
    odh.get_data(
        folder_location=dataset_folder,
        regex_filters=[
            RegexFilter(left_bound="C_", right_bound="_",        values=classes_values, description="classes"),
            RegexFilter(left_bound="R_", right_bound="_emg.csv", values=reps_values,    description="reps"),
        ],
    )

    filt = Filter(cfg.SAMPLING)
    filt.install_filters({"name": "notch",    "cutoff": 60,        "bandwidth": 3})
    filt.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
    filt.filter(odh)
    return odh


def main(argv=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import emagerlib.models.new_emager_cnn as new_models

    _, cfg = setup_runtime(argv)

    # -- 1. Load & filter raw EMG -------------------------------------------------
    odh = load_and_filter(cfg.DATASETS_PATH, cfg)

    # -- 2. Split by repetition ---------------------------------------------------
    train_odh = odh.isolate_data("reps", cfg.TRAIN_REPS)
    test_odh  = odh.isolate_data("reps", cfg.TEST_REPS)

    # -- 3. Extract windows -------------------------------------------------------
    train_windows, train_meta = train_odh.parse_windows(cfg.WINDOW_SIZE, cfg.WINDOW_INCREMENT)
    test_windows,  test_meta  = test_odh.parse_windows( cfg.WINDOW_SIZE, cfg.WINDOW_INCREMENT)
    logger.info(f"Windows  -- train: {train_windows.shape},  test: {test_windows.shape}")

    # -- 4. MAV compression (time axis -> scalar per channel) ---------------------
    # Each window (WINDOW_SIZE x 64 channels) is reduced to one value per channel.
    # MAV collapses the time axis; the CNN sees a 4x16 map of signal energy.
    train_mav    = np.mean(np.abs(train_windows), axis=2)   # (N_train, 64)
    train_labels = train_meta["classes"]
    test_mav     = np.mean(np.abs(test_windows),  axis=2)   # (N_test,  64)
    test_labels  = test_meta["classes"]

    # -- 5. Build DataLoaders (shared across all models) --------------------------
    train_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(train_mav.astype(np.float32)),
            torch.from_numpy(train_labels),
        ),
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )
    test_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(test_mav.astype(np.float32)),
            torch.from_numpy(test_labels),
        ),
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    # -- 6. Train & evaluate each model -------------------------------------------
    results   = {}
    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%Hh%M")

    for model_name in MODELS_TO_TEST:
        logger.info("-" * 50)
        logger.info(f"Training {model_name}")
        logger.info("-" * 50)

        model_class = getattr(new_models, model_name)
        model = model_class((4, 16), cfg.NUM_CLASSES)
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        res = model.fit(train_dl, test_dl, max_epochs=cfg.EPOCH)
        acc = res[0]["test_acc"]
        results[model_name] = acc
        logger.info(f"{model_name} -- test accuracy: {acc:.1%}")

        model_path = Path(cfg.SAVE_PATH) / f"{model_name}_{cfg.SESSION}_{acc:.3f}_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved: {model_path}")

    # -- 7. Summary ---------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Results summary")
    logger.info("=" * 50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name:<22} {acc:.1%}")

    return results


if __name__ == "__main__":
    main()
