
import threading
from libemg.data_handler import OnlineDataHandler
from libemg.emg_predictor import EMGClassifier, OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.streamers import emager_streamer
from libemg.filtering import Filter
from libemg.environments.controllers import ClassifierController

import models.models as etm
import utils.utils as eutils
from visualization.realtime_gui import RealTimeGestureUi
import utils.gestures_json as gjutils

import time
import torch
import numpy as np
from multiprocessing import Lock
from multiprocessing.connection import Connection
from config_emager import *

eutils.set_logging()

def update_labels_process(stop_event:threading.Event, gui:RealTimeGestureUi, conn:Connection | None = None, delay:float=0.01, timeout_delay:float=0.5):
    '''
    Update the labels of the gui and send the data to the controller via conn if it is not None
    stop_event: threading.Event = threading.Event()
    gui: RealTimeGestureUi = RealTimeGestureUi()
    conn: Connection | None = None, delay:float=0.01
    conn ouputs:
        output_data = {
            "prediction": int(predictions[0]),
            "timestamp": time.time()
        }
    '''
    gestures_dict = gjutils.get_gestures_dict(MEDIA_PATH)
    images = gjutils.get_images_list(MEDIA_PATH)
    ctrl = ClassifierController('predictions', NUM_CLASSES)
    
    # Track last prediction to avoid sending duplicates
    last_prediction = None
    last_sent_time = 0
    
    # Run thread until stop event
    while not stop_event.is_set():
        
        # Get predictions using the controller
        predictions = ctrl.get_data(['predictions'])
        # action = ctrl._get_action()
        # print(f"{predictions} (predictions)")
        # print(f"action: {action}")
        if predictions is None:
            time.sleep(delay)  # Wait a bit if no data
            continue
        
        index = int(predictions[0])
        
        # Only process and send if prediction has changed or enough time has passed
        current_time = time.time()
        if index == last_prediction and (current_time - last_sent_time) < timeout_delay:
            time.sleep(delay)
            continue
        
        last_prediction = index
        last_sent_time = current_time
        
        ts = current_time
        timestamp = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts - int(ts)) * 1000):03d}"
        output_data = {
            "prediction": index,
            "timestamp": timestamp
        }
        
        label = gjutils.get_label_from_index(index, images, gestures_dict)

        gui.update_label(label)

        if conn is not None:
            print(f"Output : pred({predictions[0]})  gest[{label}] {output_data}" + " "*10,"... sending data ...")
            conn.send(output_data)

        time.sleep(delay)
        

def predicator(use_gui:bool=True, conn:Connection | None = None, delay:float=0.01, timeout_delay:float=0.5):

    # Create data handler and streamer
    p, smi = emager_streamer()
    print(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)

    filter = Filter(SAMPLING)
    notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
    filter.install_filters(notch_filter_dictionary)
    bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
    filter.install_filters(bandpass_filter_dictionary)
    odh.install_filter(filter)
    print("Data handler created")

    # Choose feature group and classifier
    fe = FeatureExtractor()
    fg = ["MAV"]
    print("Feature group: ", fg)

    # Verify model loading and state dict compatibility
    model = etm.EmagerCNN((4, 16), NUM_CLASSES, -1)
    print("Loading model from: ", MODEL_PATH)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
    except RuntimeError as e:
        print(f"Error loading model: {e}")

    classi = EMGClassifier(model)
    classi.add_majority_vote(MAJORITY_VOTE)

    # Ensure OnlineEMGClassifier is correctly set up for data handling and inference
    smm_items=[
            ["classifier_output", (100,4), np.double, Lock()], #timestamp, class prediction, confidence, velocity
            ['classifier_input', (100, 1 + 64), np.double, Lock()], # timestamp <- features ->
        ]
    oclassi = OnlineEMGClassifier(classi, WINDOW_SIZE, WINDOW_INCREMENT, odh, fg, std_out=False, smm=True, smm_items=smm_items)


    # Create GUI
    files = gjutils.get_images_list(MEDIA_PATH)
    print("Files: ", files)
    print("Creating GUI...")
    gui = RealTimeGestureUi(files)
    
    stop_event = threading.Event()
    updateLabelProcess = threading.Thread(target=update_labels_process, args=(
        stop_event, gui, conn, delay, timeout_delay))

    try:
        print("Starting classification...")
        oclassi.run(block=False)
        print("Starting process thread...")
        updateLabelProcess.start()
        print("Starting GUI...")
        if use_gui:
            gui.run()
        else:
            while True:
                time.sleep(1)
        
    except Exception as e:
        print(f"Error during classification: {e}")

    finally :
        if conn is not None:
            conn.send("exit")
        stop_event.set()
        oclassi.stop_running()
        
        print("Exiting")


if __name__ == "__main__":
    predicator(use_gui=True, conn=None, delay=0.01, timeout_delay=0.5)