import time
from pathlib import Path
import logging

from libemg.data_handler import OnlineDataHandler
from libemg.streamers import emager_streamer
from libemg.filtering import Filter
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
parser = create_parser(
    description="Visualize EMG signals using libemg",
    default_config=str(DEFAULT_CONFIG)
)
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging (after loading config so it can use config defaults)
setup_logging(args, cfg, script_name="libemg_visualize")

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)

# Save config if requested (uses logging internally)
save_config_if_requested(args, cfg, script_name="libemg_visualize")

def main():
    # Create data handler and streamer
    p, smi = emager_streamer()
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
        
if __name__ == "__main__":
    main()