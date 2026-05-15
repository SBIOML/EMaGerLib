from emagerlib import ROOT_EMAGERLIB

# ===== Paths ===== #
BASE_PATH = ROOT_EMAGERLIB / "Datasets"
# BASE_PATH = Path("../data_collection/Datasets")
SESSION = "Felix_5sessions"
MEDIA_PATH = str(ROOT_EMAGERLIB / "media-test")
MODEL_NAME = None # Note: MODEL_NAME will find the last model if set to None but you can hardcode it
# or put your own function in the config.py file
# import emagerlib.utils.find_models as futils
# MODEL_NAME = futils.find_last_model(str(BASE_PATH), SESSION)
MODEL_NAME = "felix_5sessions.pth"

# ===== Gesture/classes configuration ===== #
# N, HC, HO, TU, OK, RO, IDX
CLASSES = [1, 2, 3, 14, 18, 19, 30]
NUM_CLASSES = len(CLASSES)
NUM_REPS = 10
REP_TIME = 3
REST_TIME = 1

# ===== Data acquisition and model parameters ===== #
MAJORITY_VOTE = 30
WINDOW_SIZE = 200
WINDOW_INCREMENT = 10
EPOCH = 10
SAMPLING = 2000
TRAIN_REPS = [0,1,2,3,4,5,6,7,8,9]
TEST_REPS = [4,5]
FILTER = False
VIRTUAL = False
PORT = None
EMAGER_VERSION = "v3.0"  # EMaGer device version ("v1.0", "v1.1", or "v3.0")

# ===== Controller and predictor settings ===== #
USE_GUI = True

# Thread speed control (2 independent variables):
PREDICTOR_DELAY = 0.01          # Thread 1: Predictor speed (seconds between prediction updates)
CONTROLLER_POLL_RATE = 0.001    # Thread 2: Controller speed (seconds between sending checks)

PREDICTOR_TIMEOUT_DELAY = 0.05  # Min time before processing same prediction again

# Smoothing configuration:
SMOOTH_WINDOW = 1  # Set to 1 to disable smoothing (always use latest value)
SMOOTH_METHOD = 'mode'  # 'mode' recommended for categorical gestures; 'mean' for numeric smoothing

# ===== Logging configuration ===== #
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = False  # Set to True to enable file logging by default
LOG_FILE_PATH = None  # Path to log file, or None for auto-generated name
LOG_FILE_NAME = None  # Custom log file name, or None for auto-generated name

# ===== Config saving defaults ===== #
SAVE_CONFIG_PATH = None  # Path to save config file, or None for default location
SAVE_CONFIG_NAME = None  # Custom config file name, or None for auto-generated name
SAVE_CONFIG_FORMAT = "json"  # json or yaml
