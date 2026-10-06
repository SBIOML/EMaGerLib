import libemg
import os
from config import *

if __name__ == "__main__":
    p, smm = libemg.streamers.emagerv3_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smm)
    # odh.analyze_hardware(10)
    # odh.visualize(num_samples=2000)

    # args={'media_folder': 'images/','data_folder': 'data/oymotion/train/','num_reps': 1,'rep_time': 3,'rest_time': 1,'auto_advance': True}
    args = {
        "online_data_handler": odh,
        "media_folder": MEDIA_PATH,
        "data_folder": DATAFOLDER,
        "num_reps": NUM_REPS,
        "rep_time": REP_TIME,
        "rest_time": REST_TIME,
        "auto_advance": True
    }
    gui = libemg.gui.GUI(odh, args=args)
    gui.download_gestures(CLASSES, MEDIA_PATH, download_gifs=False)
    dataset_path = os.path.abspath(DATAFOLDER + "/")

    # Ensure folder exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    gui.start_gui()

    # args={'media_folder': 'images/','data_folder': 'data/oymotion/day1/','num_reps': 3,'rep_time': 3,'rest_time': 1,'auto_advance': True}
    # gui = libemg.gui.GUI(odh, args=args)
    # gui.start_gui()
    #
    # args={'media_folder': 'images/','data_folder': 'data/oymotion/limbpos/','num_reps': 3,'rep_time': 3,'rest_time': 1,'auto_advance': True}
    # gui = libemg.gui.GUI(odh, args=args)
    # gui.start_gui()
    #
    # args={'media_folder': 'images/','data_folder': 'data/oymotion/shift/','num_reps': 3,'rep_time': 5,'rest_time': 1,'auto_advance': True}
    # gui = libemg.gui.GUI(odh, args=args)
    # gui.start_gui()