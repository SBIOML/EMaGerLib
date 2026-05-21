"""
Run all three CNN training scripts back-to-back with the test config
and print a final comparison of their results.

Usage:
    python run_all_tests.py
    python run_all_tests.py --config path/to/your_config.py
"""
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_all_tests")

DEFAULT_CONFIG = str(Path(__file__).parent.parent.parent / "config_examples" / "test_new_cnn_config.py")


def main():
    config = DEFAULT_CONFIG
    if "--config" in sys.argv:
        config = sys.argv[sys.argv.index("--config") + 1]

    args = ["--config", config]
    results = {}

    # ── 1. Legacy (train_cnn.py) ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("1/3  train_cnn.py  (legacy model)")
    logger.info("=" * 60)
    from examples.training.train_cnn import main as run_legacy
    run_legacy(args)

    # ── 2. Clean reference (train_cnn_v2.py) ─────────────────────────
    logger.info("=" * 60)
    logger.info("2/3  train_cnn_v2.py  (clean reference)")
    logger.info("=" * 60)
    from examples.training.train_cnn_v2 import main as run_v2
    run_v2(args)

    # ── 3. New model variants (train_cnn_new.py) ──────────────────────
    logger.info("=" * 60)
    logger.info("3/3  train_cnn_new.py  (new variants)")
    logger.info("=" * 60)
    from examples.training.train_cnn_new import main as run_new
    run_new(args)

    logger.info("All runs complete — check logs above for per-model accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
