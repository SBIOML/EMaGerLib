
# hand_close, hand_open, index extension, ok, thumbs up
CLASSES = [2,3,30,14,18]
NUM_CLASSES = 5
NUM_REPS = 5
REP_TIME = 5
REST_TIME = 1

# Data acquisition and model parameters
MAJORITY_VOTE=30
WINDOW_SIZE = 200
WINDOW_INCREMENT = 10
EPOCH = 10
SAMPLING = 1010
FILTER = False
VIRTUAL = False
PORT = None

# Controller and predictor settings
USE_GUI = False
POLL_SLEEP_DELAY = 0.001  # Sleep time to prevent busy waiting when polling for data (not delay between gestures sent to hand)
# TODO : add a way to fix the frequency of gestures sent to hand
PREDICTOR_DELAY = 0.01 # Changes the precision of frequency of predictions
PREDICTOR_TIMEOUT_DELAY = 0.50 # Timeout for waiting for new predictions (freq of predictions updates)
SMOOTH_WINDOW = 1 # Set to 1 to disable smoothing (always use latest value)
SMOOTH_METHOD = 'mode' # 'mode' recommended for categorical gestures; 'mean' for numeric smoothing
# Heartbeat interval (seconds) for re-sending the last gesture to the hand
# This is used when no new prediction data is available but you still want
# the controller to send the last known gesture regularly (keep-alive / repeat).
# Tune to desired rate (e.g. 0.1 = 10 Hz). Set to 0 or None to disable heartbeat.
HEARTBEAT_INTERVAL = 0.1

BASE_PATH = "./Datasets/"
SESSION = "D1"

import utils.find_models as futils
# MODEL_NAME = "libemg_torch_cnn_D0_974_25-10-20_15h03.pth"
MODEL_NAME = futils.find_last_model(BASE_PATH, SESSION)

MEDIA_PATH = "./media-test/"
MODEL_PATH = f"{BASE_PATH}{SESSION}/{MODEL_NAME}"
DATAFOLDER = f"{BASE_PATH}{SESSION}/"
DATASETS_PATH = f"{BASE_PATH}{SESSION}/"
SAVE_PATH = f"{BASE_PATH}{SESSION}/"