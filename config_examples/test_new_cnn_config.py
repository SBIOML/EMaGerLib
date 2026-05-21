from emagerlib import ROOT_EMAGERLIB

# ===== Paths ===== #
BASE_PATH = ROOT_EMAGERLIB / "Datasets"
SESSION   = "Test_EM_C7_R5"
MEDIA_PATH            = ROOT_EMAGERLIB / "media-test"
MODEL_NAME            = None
PRETRAINED_MODEL_PATH = ROOT_EMAGERLIB / "models" / "placeholder.pth"

# ===== Gesture/classes configuration ===== #
# 7 classes, 0-indexed — verify these match the folder names in your dataset
CLASSES  = [1, 2, 3, 14, 18, 19, 30]
NUM_REPS = 5
REP_TIME  = 3
REST_TIME = 1

# ===== Data acquisition ===== #
EMAGER_VERSION = "v3.0"
SAMPLING = 2000
FILTER   = False
VIRTUAL  = False
PORT     = None

# ===== Training parameters ===== #
TRAIN_REPS      = [0, 1, 2]
TEST_REPS       = [3, 4]
WINDOW_SIZE      = 200
WINDOW_INCREMENT = 10
EPOCH            = 10
FREEZE_FIRST_N_PARAMS = 0
USE_PRETRAINED        = False

# ===== Controller and predictor settings ===== #
USE_GUI              = False
MAJORITY_VOTE        = 30
PREDICTOR_DELAY      = 0.01
CONTROLLER_POLL_RATE = 0.001
PREDICTOR_TIMEOUT_DELAY = 0.05
SMOOTH_WINDOW = 1
SMOOTH_METHOD = "mode"

# ===== Logging configuration ===== #
LOG_LEVEL    = "INFO"
LOG_TO_FILE  = False
LOG_FILE_PATH = None
LOG_FILE_NAME = None

# ===== Config saving defaults ===== #
SAVE_CONFIG_PATH   = None
SAVE_CONFIG_NAME   = None
SAVE_CONFIG_FORMAT = "json"
