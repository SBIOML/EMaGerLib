from dataclasses import dataclass
from src.configs.base_config import BaseConfig
from pathlib import Path
import sys

@dataclass
class Config(BaseConfig):
    
    
    # SGT
    CLASSES: list = (2, 3, 10, 14, 18)
    NUM_CLASSES: int = 5
    NUM_REPS: int = 7
    GESTURE_TIME: int = 5
    REST_TIME: int = 2

    # Signal processing
    WINDOW_SIZE: int = 200
    WINDOW_INCREMENT: int = 10
    SAMPLING: int = 2000
    EPOCH: int = 25
    MAJORITY_VOTE: int = 25
    FILTER: bool = False
    VIRTUAL: bool = False
    PORT: str = None

    # Controller
    USE_GUI: bool = False
    POLL_SLEEP_DELAY: float = 0.001
    PREDICTOR_DELAY: float = 0.01
    PREDICTOR_TIMEOUT_DELAY: float = 0.50
    SMOOTH_WINDOW: int = 1
    SMOOTH_METHOD: str = 'mode'
    HEARTBEAT_INTERVAL: float = 0.005

    # Paths
    MEDIA_PATH: str = "./media-sgt/"
    BASE_PATH: str = "./Datasets/"
    SESSION: str = "Demo_IEEE"
    
    MODEL_NAME: str = None
    MODEL_PATH: str = None
    
    DATAFOLDER: str = f"{BASE_PATH}{SESSION}/"
    DATASETS_PATH: str = f"{BASE_PATH}{SESSION}/"
    SAVE_PATH: str = f"{BASE_PATH}{SESSION}/"
    
    CONFIG_FILE_NAME: str = None
    CONFIG_LOG_PATH: str = f"{BASE_PATH}{SESSION}/"
    READ_CONFIG_LOG_PATH: str = f"{BASE_PATH}{SESSION}/"

    # MODEL / FILES
    def __post_init__(self):
        try:
            from ..utils import find_models as futils
        except Exception:
            project_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(project_root.parent))
            import src.utils.find_models as futils

        # self.MODEL_NAME = "model_final.pth"
        self.MODEL_NAME = futils.find_last_model(self.BASE_PATH, self.SESSION)
        self.MODEL_PATH = f"{self.BASE_PATH}{self.SESSION}/{self.MODEL_NAME}"
        
        if self.CONFIG_FILE_NAME is None:
            self.CONFIG_FILE_NAME = f"{self.__class__.__module__.split('.')[-1]}"
