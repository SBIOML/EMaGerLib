import time
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import logging

from libemg.data_handler import OnlineDataHandler
from libemg.datasets import OneSubjectMyoDataset
from libemg.gui import GUI
from libemg.streamers import emager_streamer
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
parser = create_parser(
    description="Screen-guided EMG training data collection",
    default_config=str(DEFAULT_CONFIG)
)
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging (after loading config so it can use config defaults)
setup_logging(args, cfg, script_name="screen_guided_training")

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)

# Save config if requested (uses logging internally)
save_config_if_requested(args, cfg, script_name="screen_guided_training")


def main():
    # Create data handler and streamer
    p, smi = emager_streamer()
    logger.info(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)
    logger.info("Data handler created")

    args = {
        "online_data_handler": odh,
        "media_folder": cfg.MEDIA_PATH,
        "data_folder": cfg.DATAFOLDER,
        "num_reps": cfg.NUM_REPS,
        "rep_time": cfg.REP_TIME,
        "rest_time": cfg.REST_TIME,
        "auto_advance": True
    }
    
    gui = GUI(odh, args=args, debug=False, width=900, height=800)
    gui.download_gestures(cfg.CLASSES, cfg.MEDIA_PATH, download_gifs=False)

    dataset_path = os.path.abspath(str(cfg.DATAFOLDER))

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
                confirm = input(f"Are you sure you want to DELETE all contents of '{dataset_path}'? Type 'yes' to confirm: ").strip().lower()
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
                cfg.DATAFOLDER = candidate
                args["data_folder"] = cfg.DATAFOLDER
                # Try to update GUI-held args if possible
                try:
                    setattr(gui, "args", args)
                except Exception:
                    pass
                logger.info(f"Dataset directory renamed to '{cfg.DATAFOLDER}'. Using that path for this run.")
                break
            elif choice == "c" or choice == "cancel":
                logger.info("Operation cancelled. Exiting.")
                sys.exit(0)
            else:
                logger.warning("Invalid choice. Please enter O, R, or C.")
    else:
        # Ensure folder exists (redundant safe-guard)
        os.makedirs(dataset_path, exist_ok=True)
    
    gui.start_gui()

    logger.info(f"Data saved in: {cfg.DATAFOLDER}")
    
if __name__ == "__main__":
    main()