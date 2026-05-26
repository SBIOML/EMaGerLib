"""
Compare Legacy, Reference, and Base EmagerCNN on the same dataset.

Calls each training script main() in sequence so the full pipeline of each
script runs as-is. Only EmagerCNNBase is run from train_cnn_benchmark so
results are directly comparable (same architecture as Reference).

  Legacy    -- train_cnn.py           (Conv -> ReLU -> BN, non-standard)
  Reference -- train_cnn_v2.py        (Conv -> BN -> ReLU, standard)
  Base      -- train_cnn_benchmark.py (Conv -> BN -> ReLU, same as Reference)

If Reference == Base: the two codebases are equivalent.
If Legacy differs:    it is due to the non-standard activation order.

Usage:
    python train_cnn_compare.py
    python train_cnn_compare.py --config path/to/config.py
"""
from pathlib import Path
import sys
import logging
import random

import lightning
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_cnn_compare")

DEFAULT_CONFIG = str(Path(__file__).parent.parent.parent / "config_examples" / "test_new_cnn_config.py")

# Single seed for a quick equivalence check; extend (e.g. [42, 123, 456]) for statistics.
SEEDS = [42]


def main(seeds=None):
    config = DEFAULT_CONFIG
    if "--config" in sys.argv:
        config = sys.argv[sys.argv.index("--config") + 1]

    args = ["--config", config]
    _seeds = seeds if seeds is not None else SEEDS

    from emagerlib.config.load_config import load_config as _load_config
    _cfg = _load_config(config)

    from examples.training.train_cnn    import main as run_legacy
    from examples.training.train_cnn_v2 import main as run_reference
    import examples.training.train_cnn_benchmark as bm

    # -- 1. Legacy (train_cnn.py -- Conv -> ReLU -> BN) -----------------------
    logger.info("=" * 60)
    logger.info("1/3  train_cnn.py  (Legacy -- Conv -> ReLU -> BN)")
    logger.info("=" * 60)
    accs_legacy = []
    for seed in _seeds:
        lightning.seed_everything(seed)
        accs_legacy.append(run_legacy(args))

    # -- 2. Reference (train_cnn_v2.py -- Conv -> BN -> ReLU) -----------------
    logger.info("=" * 60)
    logger.info("2/3  train_cnn_v2.py  (Reference -- Conv -> BN -> ReLU)")
    logger.info("=" * 60)
    accs_reference = []
    for seed in _seeds:
        lightning.seed_everything(seed)
        accs_reference.append(run_reference(args))

    # -- 3. Base (train_cnn_benchmark.py -- EmagerCNNBase only) ---------------
    logger.info("=" * 60)
    logger.info("3/3  train_cnn_benchmark.py -> EmagerCNNBase  (Conv -> BN -> ReLU)")
    logger.info("=" * 60)
    results_base = bm.main(config=config, datasets=[_cfg.SESSION], seeds=_seeds, models=["EmagerCNNBase"])
    accs_base = results_base["EmagerCNNBase"][_cfg.SESSION]

    # -- Final comparison --------------------------------------------------------
    mean_legacy = np.mean(accs_legacy);    std_legacy = np.std(accs_legacy)
    mean_ref    = np.mean(accs_reference); std_ref    = np.std(accs_reference)
    mean_base   = np.mean(accs_base);     std_base   = np.std(accs_base)

    logger.info("=" * 60)
    logger.info(f"Accuracy comparison  (seeds: {_seeds})")
    logger.info("=" * 60)
    logger.info(f"  Legacy    (Conv->ReLU->BN)  {mean_legacy:.1%} +/- {std_legacy:.1%}")
    logger.info(f"  Reference (Conv->BN->ReLU)  {mean_ref:.1%} +/- {std_ref:.1%}")
    logger.info(f"  Base      (Conv->BN->ReLU)  {mean_base:.1%} +/- {std_base:.1%}")
    logger.info("-" * 60)
    delta_ref_base = abs(mean_ref - mean_base)
    logger.info(f"  Reference vs Base delta: {delta_ref_base:.2%}")
    if delta_ref_base < 0.005:
        logger.info("  -> Reference and Base are equivalent (delta < 0.5%)")
    else:
        logger.info("  -> Reference and Base diverge -- investigate.")

    return accs_legacy, accs_reference, accs_base


if __name__ == "__main__":
    main()
