from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from emagerlib.utils.find_models import find_last_model


@dataclass
class CoreConfig:
    """Core configuration for EMG recording sessions."""
    
    # ===== PATHS ===== #
    BASE_PATH: Path
    SESSION: str
    MEDIA_PATH: Path
    MODEL_NAME: str
    PRETRAINED_MODEL_PATH: Path
    
    # ===== GESTURE/CLASSES CONFIGURATION ===== #
    CLASSES: List[int]
    NUM_REPS: int
    REP_TIME: int
    REST_TIME: int

    # ===== DATA ACQUISITION ===== #
    EMAGER_VERSION: str
    SAMPLING: int
    FILTER: bool
    VIRTUAL: bool
    PORT: Optional[str]

    # ===== TRAINING PARAMETERS ===== #
    TRAIN_REPS: List[int]
    TEST_REPS: List[int]
    WINDOW_SIZE: int
    WINDOW_INCREMENT: int
    EPOCH: int
    FREEZE_FIRST_N_PARAMS: int
    USE_PRETRAINED: bool

    # ===== CONTROLLER AND PREDICTOR SETTINGS ===== #
    USE_GUI: bool
    MAJORITY_VOTE: int
    PREDICTOR_DELAY: float
    CONTROLLER_POLL_RATE: float
    PREDICTOR_TIMEOUT_DELAY: float
    SMOOTH_WINDOW: int
    SMOOTH_METHOD: str

    # ===== LOGGING CONFIGURATION ===== #
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: Optional[Path] = None
    LOG_FILE_NAME: Optional[str] = None

    # ===== CONFIG SAVING DEFAULTS ===== #
    SAVE_CONFIG_PATH: Optional[Path] = None
    SAVE_CONFIG_NAME: Optional[str] = None
    SAVE_CONFIG_FORMAT: str = "json"

    # ===== EXTRA VALUES ADDED LATER ===== #
    EXTRA: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # ===== OPTIONAL METADATA COMPUTED LATER ===== #
    CONFIG_FILE_NAME: Optional[str] = None
    CONFIG_FILE_PATH: Optional[str] = None

    # ===== COMPUTED PROPERTIES ===== #
    @property
    def NUM_CLASSES(self) -> int:
        return len(self.CLASSES)
    
    @property
    def SESSION_PATH(self) -> Path:
        return self.BASE_PATH / self.SESSION

    @property
    def SAVE_PATH(self) -> Path:
        return self.SESSION_PATH
    
    @property
    def DATAFOLDER(self) -> Path:
        return self.SESSION_PATH
    
    @property
    def DATASETS_PATH(self) -> Path:
        return self.SESSION_PATH
    
    @property
    def MODEL_PATH(self) -> Optional[Path]:
        if self.MODEL_NAME is None:
            # Auto-detect last model
            try:
                model_name = find_last_model(str(self.BASE_PATH), self.SESSION)
                if model_name:
                    return self.SESSION_PATH / model_name
            except (ImportError, Exception):
                pass
            return None
        return self.SESSION_PATH / self.MODEL_NAME

    # ===== EXTRA FIELD ACCESS (DO NOT CHANGE) ===== #
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return object.__getattribute__(self, 'EXTRA').get(name, None)
        except AttributeError:
            return None
    
    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, '__dataclass_fields__') and name in self.__dataclass_fields__:
            object.__setattr__(self, name, value)
        elif name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            if not hasattr(self, 'EXTRA'):
                object.__setattr__(self, 'EXTRA', {})
            self.EXTRA[name] = value
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)