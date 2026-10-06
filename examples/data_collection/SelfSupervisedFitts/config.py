import os

# Level 1 gestures: 1,2,3,4,5,6,7,8,14,26,30
# CLASSES = [2,3,10,14,18]
# CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 14, 26, 30] #Felix SGT
# CLASSES = [1,2,3,4,5] # Tobi SGT
CLASSES = [1, 2, 3, 4, 5]  # Quick test SGT
NUM_CLASSES = len(CLASSES)
NUM_REPS = 3
REP_TIME = 2
REST_TIME = 0

SENSOR = "EMaGer"  # "EMaGer" or "Myo"

WINDOW_SIZE = 200
WINDOW_INCREMENT = 10
EPOCH = 10
MAJORITY_VOTE = 7
SAMPLING = 2000
FILTER = False
VIRTUAL = False
PORT = None

# Sessions are saved next to this file, so the project works from any location
BASE_PATH = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
SESSION = "NEW_test_transition_sgt_2"


MEDIA_PATH = f"{BASE_PATH}media-sgt/"  # gesture images are auto-downloaded here
DATAFOLDER = f"{BASE_PATH}{SESSION}/"
DATASETS_PATH = f"{BASE_PATH}{SESSION}/"
SAVE_PATH = f"{BASE_PATH}{SESSION}/"