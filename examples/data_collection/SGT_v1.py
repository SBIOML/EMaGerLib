import time
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import logging

from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.utils.notch import get_notch_freq, save_notch_freq
from emagerlib.utils.arm import get_arm, save_arm
from libemg.streamers import emager_streamer

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Screen-guided EMG training data collection",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="screen_guided_training")
    save_config_if_requested(args, cfg, script_name="screen_guided_training")
    return args, cfg


def main(argv=None):
    from libemg.data_handler import OnlineDataHandler
    from libemg.gui import GUI

    _, cfg = setup_runtime(argv)
    media_folder = os.path.join(str(cfg.MEDIA_PATH), "")

    # Create data handler and streamer
    p, smi = emager_streamer(cfg.EMAGER_VERSION)
    logger.info(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)
    logger.info("Data handler created")

    # cfg.DATAFOLDER is a read-only property, so the (possibly renamed) output folder is tracked here
    data_folder = str(cfg.DATAFOLDER)

    gui_args = {
        "online_data_handler": odh,
        "media_folder": media_folder,
        "data_folder": data_folder,
        "num_reps": cfg.NUM_REPS,
        "rep_time": cfg.REP_TIME,
        "rest_time": cfg.REST_TIME,
        "auto_advance": True
    }

    gui = GUI(odh, args=gui_args, debug=False, width=900, height=800)
    gui.download_gestures(cfg.CLASSES, media_folder, download_gifs=False)

    dataset_path = os.path.abspath(data_folder)

    # Ensure folder exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    # Check if folder has any files/subfolders
    has_data = any(os.scandir(dataset_path))

    if has_data:
        logger.info(f"Data found in '{dataset_path}'.")
        while True:
            choice = input("Choose action - [O]verwrite, [R]ename, [C]ancel: ").strip().lower()
            if choice == "o" or choice == "overwrite":
                confirm = input(
                    f"Are you sure you want to DELETE all contents of '{dataset_path}'? Type 'yes' to confirm: ").strip().lower()
                if confirm == "yes":
                    shutil.rmtree(dataset_path)
                    os.makedirs(dataset_path, exist_ok=True)
                    logger.info(f"Contents of '{dataset_path}' deleted.")
                    break
                else:
                    logger.info("Overwrite cancelled.")
                    continue
            elif choice == "r" or choice == "rename":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_path = f"{dataset_path}_{timestamp}"
                # Ensure unique name
                i = 1
                candidate = new_path
                while os.path.exists(candidate):
                    candidate = f"{new_path}_{i}"
                    i += 1
                os.makedirs(candidate, exist_ok=True)
                data_folder = candidate
                gui_args["data_folder"] = data_folder  # the GUI reads this same dict
                logger.info(f"Dataset directory renamed to '{data_folder}'. Using that path for this run.")
                break
            elif choice == "c" or choice == "cancel":
                logger.info("Operation cancelled. Exiting.")
                sys.exit(0)
            else:
                logger.warning("Invalid choice. Please enter O, R, or C.")
    else:
        # Ensure folder exists (redundant safe-guard)
        os.makedirs(dataset_path, exist_ok=True)

    get_notch_freq(cfg)  # stop now if NOTCH_FREQ is missing or invalid, not after recording
    get_arm(cfg)         # same for ARM (left / right)
    gui.start_gui()
    # libemg writes collection_details.json when the collection starts: add the notch frequency and the arm
    save_notch_freq(data_folder, cfg)
    save_arm(data_folder, cfg)

    logger.info(f"Data saved in: {data_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())