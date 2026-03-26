
import threading
import time
import torch
import numpy as np
from multiprocessing import Lock
from multiprocessing.connection import Connection
from pathlib import Path
import logging
import sys
import os
from libemg.data_handler import OnlineDataHandler
from libemg.emg_predictor import EMGClassifier, OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.filtering import Filter
from libemg.environments.controllers import ClassifierController

import emagerlib.models.models as etm
import emagerlib.utils.utils as eutils
from emagerlib.visualization.realtime_gui import RealTimeGestureUi
import emagerlib.utils.gestures_json as gjutils
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.utils.streamer_utils import get_emager_streamer

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
parser = create_parser(
    description="Real-time EMG gesture prediction",
    default_config=str(DEFAULT_CONFIG)
)
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging (after loading config so it can use config defaults)
setup_logging(args, cfg, script_name="realtime_prediction")

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)

# Save config if requested (uses logging internally)
save_config_if_requested(args, cfg, script_name="realtime_prediction")

def update_labels_process(stop_event:threading.Event, gui:RealTimeGestureUi, ctrl:ClassifierController, conn:Connection | None = None, delay:float=cfg.PREDICTOR_DELAY, timeout_delay:float=cfg.PREDICTOR_TIMEOUT_DELAY):
    '''
    Update the labels of the gui and send the data to the controller via conn if it is not None
    stop_event: threading.Event = threading.Event()
    gui: RealTimeGestureUi = RealTimeGestureUi()
    ctrl: ClassifierController = shared classifier controller (avoid duplicate socket binding)
    conn: Connection | None = None, delay:float=0.01
    conn ouputs:
        output_data = {
            "prediction": int(predictions[0]),
            "timestamp": time.time()
        }
    '''
    gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
    images = gjutils.get_images_list(cfg.MEDIA_PATH)
    
    # Track last prediction to avoid sending duplicate data on pipe
    last_prediction = None
    last_sent_time = 0
    
    # Run thread until stop event
    while not stop_event.is_set():
        try:
            # Get predictions using the controller
            predictions = ctrl.get_data(['predictions'])
            # action = ctrl._get_action()
            # print(f"{predictions} (predictions)")
            # print(f"action: {action}")
            if predictions is None:
                time.sleep(delay)  # Wait a bit if no data
                continue
            
            index = int(predictions[0])
            
            # Validate prediction index
            if index < 0 or index >= cfg.NUM_CLASSES:
                logger.warning(f"Invalid prediction index {index}, skipping")
                time.sleep(delay)
                continue
            
            # Throttle duplicate forwarding (pipe/log), but keep GUI responsive
            current_time = time.time()
            should_forward = not (
                index == last_prediction and (current_time - last_sent_time) < timeout_delay
            )
            
            ts = current_time
            timestamp = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts - int(ts)) * 1000):03d}"
            output_data = {
                "prediction": index,
                "timestamp": timestamp
            }
            
            label = gjutils.get_label_from_index(index, images, gestures_dict)
            
            if label is None:
                logger.warning(f"Failed to get label for index {index}")
                time.sleep(delay)
                continue

            gui.update_label(label)
            
            if should_forward:
                logger.info(f"Prediction: {index} ({label})")

            if conn is not None and should_forward:
                logger.info(f"Output : pred({predictions[0]})  gest[{label}] {output_data}... sending data ...")
                conn.send(output_data)
                last_prediction = index
                last_sent_time = current_time

            if conn is None and should_forward:
                last_prediction = index
                last_sent_time = current_time

            time.sleep(delay)
        
        except Exception as e:
            logger.error(f"Error in update_labels_process loop: {e}", exc_info=True)
            time.sleep(delay)
        

def predicator(use_gui:bool=True, ctrl_shared:ClassifierController | None = None, conn:Connection | None = None, delay:float=cfg.PREDICTOR_DELAY, timeout_delay:float=cfg.PREDICTOR_TIMEOUT_DELAY):

    # Create data handler and streamer
    p, smi = get_emager_streamer(cfg.EMAGER_VERSION)
    smi_emg = [item for item in smi if item[0] in ['emg', 'emg_count']]
    logger.info(f"Streamer created: process: {p}, smi : {smi_emg}")
    odh = OnlineDataHandler(shared_memory_items=smi_emg)

    filter = Filter(cfg.SAMPLING)
    notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
    filter.install_filters(notch_filter_dictionary)
    bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
    filter.install_filters(bandpass_filter_dictionary)
    odh.install_filter(filter)
    logger.info("Data handler created")

    # Choose feature group and classifier
    fe = FeatureExtractor()
    fg = ["MAV"]
    logger.info(f"Feature group: {fg}")

    # Verify model loading and state dict compatibility
    model = etm.EmagerCNN((4, 16), cfg.NUM_CLASSES, -1)
    
    if cfg.MODEL_PATH is None:
        logger.error(f"No model found in {cfg.SESSION_PATH}. Please train a model first or specify MODEL_NAME in config.")
        logger.error(f"You can train a model using: emager-train-cnn -c <config_file>")
        sys.exit(1)
    
    logger.info(f"Loading model from: {cfg.MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(cfg.MODEL_PATH))
        model.eval()
    except RuntimeError as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {cfg.MODEL_PATH}")
        sys.exit(1)

    classi = EMGClassifier(model)
    classi.add_majority_vote(cfg.MAJORITY_VOTE)

    # Ensure OnlineEMGClassifier is correctly set up for data handling and inference
    smm_items=[
            ["classifier_output", (100,4), np.double, Lock()], #timestamp, class prediction, confidence, velocity
            ['classifier_input', (100, 1 + 64), np.double, Lock()], # timestamp <- features ->
        ]
    oclassi = OnlineEMGClassifier(classi, cfg.WINDOW_SIZE, cfg.WINDOW_INCREMENT, odh, fg, std_out=False, smm=True, smm_items=smm_items)


    # Create GUI
    files = gjutils.get_images_list(cfg.MEDIA_PATH + os.sep)
    logger.debug(f"Files: {files}")
    logger.info("Creating GUI...")
    gui = RealTimeGestureUi(files)
    
    # Use shared classifier controller if provided; otherwise create one (for backward compatibility)
    if ctrl_shared is not None:
        ctrl = ctrl_shared
        logger.debug("Using shared ClassifierController")
    else:
        ctrl = ClassifierController('predictions', cfg.NUM_CLASSES)
        logger.debug("Created new ClassifierController (standalone mode)")
    
    stop_event = threading.Event()
    updateLabelProcess = threading.Thread(target=update_labels_process, args=(
        stop_event, gui, ctrl, conn, delay, timeout_delay))

    try:
        logger.info("Starting classification...")
        oclassi.run(block=False)
        logger.info("Starting process thread...")
        updateLabelProcess.start()
        logger.info("Starting GUI...")
        if use_gui:
            gui.run()
        else:
            while True:
                time.sleep(1)
        
    except Exception as e:
        logger.error(f"Error during classification: {e}")

    finally :
        if conn is not None:
            conn.send("exit")
        stop_event.set()
        oclassi.stop_running()
        
        logger.info("Exiting")


def main():
    predicator(use_gui=True, ctrl_shared=None, conn=None, delay=cfg.PREDICTOR_DELAY, timeout_delay=cfg.PREDICTOR_TIMEOUT_DELAY)

if __name__ == "__main__":
    main()