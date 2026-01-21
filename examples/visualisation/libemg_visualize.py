import time
from pathlib import Path

from libemg.data_handler import OnlineDataHandler
from libemg.streamers import emager_streamer
from libemg.filtering import Filter
from emager_tools.config.loader import load_py_config

# Load configuration
cfg = load_py_config(Path(__file__).parent.parent / "config_examples" / "base_config_example.py")

def main():
    # Create data handler and streamer
    p, smi = emager_streamer()
    print(f"Streamer created: process: {p}, smi : {smi}")
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
        print(e)
    finally:
        print("Exiting...")