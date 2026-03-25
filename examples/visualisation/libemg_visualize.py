import time
from pathlib import Path
import logging

from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.utils.streamer_utils import get_emager_streamer

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Visualize EMG signals using libemg",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="libemg_visualize")
    save_config_if_requested(args, cfg, script_name="libemg_visualize")
    return args, cfg


def main(argv=None):
    from libemg.data_handler import OnlineDataHandler
    from libemg.filtering import Filter

    _, cfg = setup_runtime(argv)

    # Create data handler and streamer
    p, smi = get_emager_streamer(cfg.EMAGER_VERSION)
    logger.info(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)

    if cfg.FILTER:
        filter = Filter(cfg.SAMPLING)
        notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
        filter.install_filters(notch_filter_dictionary)
        bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
        filter.install_filters(bandpass_filter_dictionary)
        odh.install_filter(filter)

    try :
        odh.visualize(num_samples=5000, block=True)
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("Exiting...")
    return 0
        
if __name__ == "__main__":
    raise SystemExit(main())