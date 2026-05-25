from pathlib import Path
import json
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

# -- Configuration -----------------------------------------------------------
# Each dataset folder must contain a collection_details.json.
# num_classes and num_reps are read from that file automatically.
# Train split: reps [0 .. num_reps-2]   Test split: rep [num_reps-1]

BASE_PATH = Path(__file__).parent.parent.parent / "Datasets"

DATASETS = [
    "Test_EM_C7_R5",
    "Test_EM_C7_R5_02",
    "Test_EM_C7_R5_03",
]

# Each model is trained once per seed. Results reported as mean +/- std.
# Set to a single value (e.g. [42]) for a quick one-shot run.
SEEDS = [42, 123, 456]

# Comment / uncomment to select which variants to include.
MODELS_TO_TEST = [
    "EmagerCNNBase",
    # "EmagerCNNWide",
    # "EmagerCNNDeep",
    # "EmagerCNNLight",
    "EmagerCNNStrided",
    "EmagerCNNCircular",
    "EmagerCNNRingStrided",
    # "EmagerCNNGAP",
]

WINDOW_SIZE      = 200
WINDOW_INCREMENT = 10
MAX_EPOCHS       = 10
BATCH_TRAIN      = 64
BATCH_TEST       = 256
SAMPLING         = 2000

# ----------------------------------------------------------------------------


def read_dataset_info(dataset_path: Path):
    cj = dataset_path / "collection_details.json"
    if not cj.exists():
        raise FileNotFoundError(f"No collection_details.json in {dataset_path}")
    d = json.loads(cj.read_text())
    return d["num_motions"], d["num_reps"]


def load_and_filter(dataset_path: Path, num_classes: int, num_reps: int, sampling: int):
    """Load raw EMG from disk and apply notch + bandpass filters."""
    from libemg.data_handler import OfflineDataHandler, RegexFilter
    from libemg.filtering import Filter

    odh = OfflineDataHandler()
    odh.get_data(
        folder_location=str(dataset_path),
        regex_filters=[
            RegexFilter(left_bound="C_", right_bound="_",        values=[str(i) for i in range(num_classes)], description="classes"),
            RegexFilter(left_bound="R_", right_bound="_emg.csv", values=[str(i) for i in range(num_reps)],    description="reps"),
        ],
    )
    filt = Filter(sampling)
    filt.install_filters({"name": "notch",    "cutoff": 60,        "bandwidth": 3})
    filt.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
    filt.filter(odh)
    return odh


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_results_table(results: dict, datasets: list, models: list, seeds: list):
    col_w = 22
    ds_w  = 20
    ov_w  = 14

    overall = {}
    for m in models:
        all_accs = [a for d in datasets for a in results[m].get(d, [])]
        overall[m] = np.mean(all_accs) if all_accs else float("nan")

    header = "Model".ljust(col_w) + "".join(d[:ds_w].center(ds_w) for d in datasets) + "Overall".center(ov_w)
    sep    = "-" * len(header)

    logger.info("=" * len(header))
    logger.info(f"Benchmark results  (seeds: {seeds})")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info(sep)

    for m in sorted(models, key=lambda x: overall[x], reverse=True):
        row = m.ljust(col_w)
        for d in datasets:
            accs = results[m].get(d, [])
            cell = f"{np.mean(accs):.1%}+/-{np.std(accs):.1%}" if accs else "N/A"
            row += cell.center(ds_w)
        ov_str = f"{overall[m]:.1%}" if not np.isnan(overall[m]) else "N/A"
        row += ov_str.center(ov_w)
        logger.info(row)

    logger.info(sep)


def main(datasets=None, seeds=None, models=None, config=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import emagerlib.models.new_emager_cnn as new_models

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # If a config file is provided, pull training hyperparams from it.
    # Dataset selection is always controlled by the datasets param / DATASETS constant.
    if config is not None:
        from emagerlib.config.load_config import load_config as _load_config
        _cfg             = _load_config(config)
        _window_size      = _cfg.WINDOW_SIZE
        _window_increment = _cfg.WINDOW_INCREMENT
        _max_epochs       = _cfg.EPOCH
        _sampling         = _cfg.SAMPLING
    else:
        _window_size      = WINDOW_SIZE
        _window_increment = WINDOW_INCREMENT
        _max_epochs       = MAX_EPOCHS
        _sampling         = SAMPLING

    _datasets = datasets if datasets is not None else DATASETS
    _seeds    = seeds    if seeds    is not None else SEEDS
    _models   = models   if models   is not None else MODELS_TO_TEST

    # results[model][dataset] = [acc_seed0, acc_seed1, ...]
    results = {m: {} for m in _models}

    for dataset_name in _datasets:
        dataset_path = BASE_PATH / dataset_name
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info("=" * 60)

        try:
            num_classes, num_reps = read_dataset_info(dataset_path)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset_name}: {e}")
            continue

        logger.info(f"  {num_classes} classes, {num_reps} reps -- train=[0..{num_reps-2}], test=[{num_reps-1}]")

        # -- 1. Load & filter raw EMG -----------------------------------------
        odh = load_and_filter(dataset_path, num_classes, num_reps, _sampling)

        # -- 2. Split by repetition -------------------------------------------
        train_odh = odh.isolate_data("reps", list(range(num_reps - 1)))
        test_odh  = odh.isolate_data("reps", [num_reps - 1])

        # -- 3. Extract windows -----------------------------------------------
        train_windows, train_meta = train_odh.parse_windows(_window_size, _window_increment)
        test_windows,  test_meta  = test_odh.parse_windows( _window_size, _window_increment)
        logger.info(f"  Windows -- train: {train_windows.shape},  test: {test_windows.shape}")

        # -- 4. MAV compression (time axis -> scalar per channel) -------------
        # MAV collapses the time axis; the CNN sees a 4x16 map of signal energy.
        train_mav    = np.mean(np.abs(train_windows), axis=2)   # (N, 64)
        train_labels = train_meta["classes"]
        test_mav     = np.mean(np.abs(test_windows),  axis=2)   # (N, 64)
        test_labels  = test_meta["classes"]

        for model_name in _models:
            logger.info("-" * 50)
            logger.info(f"  {model_name}")
            logger.info("-" * 50)
            seed_accs = []

            for seed in _seeds:
                # -- 5. Build DataLoaders (rebuilt per seed for deterministic shuffle) --
                set_seed(seed)
                train_dl = DataLoader(
                    TensorDataset(
                        torch.from_numpy(train_mav.astype(np.float32)),
                        torch.from_numpy(train_labels),
                    ),
                    batch_size=BATCH_TRAIN, shuffle=True, drop_last=True,
                )
                test_dl = DataLoader(
                    TensorDataset(
                        torch.from_numpy(test_mav.astype(np.float32)),
                        torch.from_numpy(test_labels),
                    ),
                    batch_size=BATCH_TEST, shuffle=False, drop_last=False,
                )

                # -- 6. Train & evaluate --------------------------------------
                model_class = getattr(new_models, model_name)
                model = model_class((4, 16), num_classes)
                logger.info(f"    seed={seed}  params={sum(p.numel() for p in model.parameters()):,}")

                res = model.fit(train_dl, test_dl, max_epochs=_max_epochs)
                acc = res[0]["test_acc"]
                seed_accs.append(acc)
                logger.info(f"    seed={seed}  acc={acc:.1%}")

            results[model_name][dataset_name] = seed_accs
            mean = np.mean(seed_accs)
            std  = np.std(seed_accs)
            logger.info(f"  {model_name:<22} {mean:.1%} +/- {std:.1%}")

    # -- 7. Print results table -----------------------------------------------
    print_results_table(results, _datasets, _models, _seeds)
    return results


if __name__ == "__main__":
    import sys
    _config = None
    if "--config" in sys.argv:
        _config = sys.argv[sys.argv.index("--config") + 1]
    main(config=_config)
