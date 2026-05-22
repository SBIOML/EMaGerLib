"""
Compare Legacy, Reference, and Base EmagerCNN on the same dataset.

Calls each training script main() in sequence, exactly like test_all_train_cnn.py did,
so the full pipeline of each script runs as-is. Only EmagerCNNBase is run from train_cnn_new
so results are directly comparable (same architecture as Reference).

  Legacy    -- train_cnn.py     (Conv -> ReLU -> BN, non-standard)
  Reference -- train_cnn_v2.py  (Conv -> BN -> ReLU, standard)
  Base      -- train_cnn_new.py (Conv -> BN -> ReLU, same as Reference)

If Reference == Base: the two codebases are equivalent.
If Legacy differs:    it is due to the non-standard activation order.

Usage:
    python train_cnn_compare.py
    python train_cnn_compare.py --config path/to/config.py
"""
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_cnn_compare")

DEFAULT_CONFIG = str(Path(__file__).parent.parent.parent / "config_examples" / "test_new_cnn_config.py")


def main():
    config = DEFAULT_CONFIG
    if "--config" in sys.argv:
        config = sys.argv[sys.argv.index("--config") + 1]

    args = ["--config", config]

    # -- 1. Legacy (train_cnn.py -- Conv -> ReLU -> BN) -----------------------
    logger.info("=" * 60)
    logger.info("1/3  train_cnn.py  (Legacy -- Conv -> ReLU -> BN)")
    logger.info("=" * 60)
    from examples.training.train_cnn import main as run_legacy
    acc_legacy = run_legacy(args)

    # -- 2. Reference (train_cnn_v2.py -- Conv -> BN -> ReLU) -----------------
    logger.info("=" * 60)
    logger.info("2/3  train_cnn_v2.py  (Reference -- Conv -> BN -> ReLU)")
    logger.info("=" * 60)
    from examples.training.train_cnn_v2 import main as run_reference
    acc_reference = run_reference(args)

    # -- 3. Base (train_cnn_new.py -- EmagerCNNBase only) ---------------------
    logger.info("=" * 60)
    logger.info("3/3  train_cnn_new.py -> EmagerCNNBase  (Conv -> BN -> ReLU)")
    logger.info("=" * 60)
    import examples.training.train_cnn_new as new_script
    new_script.MODELS_TO_TEST = ["EmagerCNNBase"]
    results_base = new_script.main(args)
    acc_base = results_base["EmagerCNNBase"]

    # -- Final comparison --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Accuracy comparison")
    logger.info("=" * 60)
    logger.info(f"  Legacy    (Conv->ReLU->BN)  {acc_legacy:.1%}")
    logger.info(f"  Reference (Conv->BN->ReLU)  {acc_reference:.1%}")
    logger.info(f"  Base      (Conv->BN->ReLU)  {acc_base:.1%}")
    logger.info("-" * 60)
    delta_ref_base = abs(acc_reference - acc_base)
    logger.info(f"  Reference vs Base delta: {delta_ref_base:.2%}")
    if delta_ref_base < 0.005:
        logger.info("  -> Reference and Base are equivalent (delta < 0.5%)")
    else:
        logger.info("  -> Reference and Base diverge -- investigate.")

    return 0


if __name__ == "__main__":
    main()
