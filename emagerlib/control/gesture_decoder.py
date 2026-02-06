from emagerlib.control.constants import *
import emagerlib.utils.gestures_json as gj
from libemg.gui import GUI
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level configuration - set via setup_gesture_decoder()
_MEDIA_PATH = None
_gestures_dict = None
_is_initialized = False

# FINGER POSITIONS PREDEFINED
NUTRAL_FINGER_POS = 250
OPEN_FIGER_POS = 0
CLOSE_FIGER_POS = 1000
HALF_OPEN_FIGER_POS = 500
FRONT_ROTATION_POS = -1000
HALF_ROTATION_POS = -500
SIDE_ROTATION_POS = 0

def setup_gesture_decoder(cfg):
    """
    Setup gesture decoder with configuration. Call this once during initialization.
    Downloads gestures if not already present.
    
    Args:
        cfg: Configuration object with MEDIA_PATH attribute
    """
    global _MEDIA_PATH, _gestures_dict, _is_initialized
    
    _MEDIA_PATH = str(cfg.MEDIA_PATH)
    
    logger.info(f"Setting up gesture decoder with media path: {_MEDIA_PATH}")
    
    try:
        _gestures_dict = gj.get_gestures_dict(_MEDIA_PATH)
        logger.info(f"Gestures loaded from {_MEDIA_PATH}")
    except FileNotFoundError:
        logger.info(f"Gestures not found, downloading to {_MEDIA_PATH}...")
        GUI.download_gestures(_MEDIA_PATH)
        _gestures_dict = gj.get_gestures_dict(_MEDIA_PATH)
        logger.info(f"Gestures downloaded successfully to {_MEDIA_PATH}")
    
    _is_initialized = True

def _ensure_initialized():
    """Ensure gesture decoder is initialized, raises error if not setup."""
    global _is_initialized
    
    if not _is_initialized:
        raise RuntimeError(
            "Gesture decoder not initialized. "
            "Call setup_gesture_decoder(cfg) before using decode_gesture(). "
            "This should be done in InterfaceControl initialization."
        )

def decode_gesture(gesture):
    """
    Decode a gesture to finger positions.
    
    Args:
        gesture: Gesture ID (int) or name (str)
        
    Returns:
        Tuple of (thumb, index, middle, ring, little, thumb_rotation) positions
    """
    _ensure_initialized()
    
    if isinstance(gesture, int):
        gestures_name = _gestures_dict[str(gesture)]
    if isinstance(gesture, str):
        gestures_name = gesture
        gesture = int(list(_gestures_dict.keys())[list(_gestures_dict.values()).index(gestures_name)])
    logger.info(f"Gesture: {gestures_name} ({gesture})")

    thumb_finger_pos = NUTRAL_FINGER_POS
    index_finger_pos = NUTRAL_FINGER_POS
    middle_finger_pos = NUTRAL_FINGER_POS
    ring_finger_pos = NUTRAL_FINGER_POS
    little_finger_pos = NUTRAL_FINGER_POS
    thumb_rotation_pos = SIDE_ROTATION_POS

    if gesture == Gesture.NO_MOTION:
        thumb_finger_pos = NUTRAL_FINGER_POS
        index_finger_pos = NUTRAL_FINGER_POS
        middle_finger_pos = NUTRAL_FINGER_POS
        ring_finger_pos = NUTRAL_FINGER_POS
        little_finger_pos = NUTRAL_FINGER_POS
        thumb_rotation_pos = SIDE_ROTATION_POS

    elif gesture == Gesture.HAND_CLOSE:
        thumb_finger_pos = HALF_OPEN_FIGER_POS
        index_finger_pos = CLOSE_FIGER_POS
        middle_finger_pos = CLOSE_FIGER_POS
        ring_finger_pos = CLOSE_FIGER_POS
        little_finger_pos = CLOSE_FIGER_POS
        thumb_rotation_pos = FRONT_ROTATION_POS

    elif gesture == Gesture.HAND_OPEN:
        thumb_finger_pos = OPEN_FIGER_POS
        index_finger_pos = OPEN_FIGER_POS
        middle_finger_pos = OPEN_FIGER_POS
        ring_finger_pos = OPEN_FIGER_POS
        little_finger_pos = OPEN_FIGER_POS
        thumb_rotation_pos = SIDE_ROTATION_POS

    elif gesture == Gesture.OK:
        thumb_finger_pos = HALF_OPEN_FIGER_POS
        index_finger_pos = HALF_OPEN_FIGER_POS
        middle_finger_pos = OPEN_FIGER_POS
        ring_finger_pos = OPEN_FIGER_POS
        little_finger_pos = OPEN_FIGER_POS
        thumb_rotation_pos = FRONT_ROTATION_POS

    elif gesture == Gesture.INDEX_EXTENSION:
        thumb_finger_pos = CLOSE_FIGER_POS
        index_finger_pos = OPEN_FIGER_POS
        middle_finger_pos = CLOSE_FIGER_POS
        ring_finger_pos = CLOSE_FIGER_POS
        little_finger_pos = CLOSE_FIGER_POS
        thumb_rotation_pos = HALF_ROTATION_POS

    elif gesture == Gesture.PEACE:
        thumb_finger_pos = NUTRAL_FINGER_POS
        index_finger_pos = OPEN_FIGER_POS
        middle_finger_pos = OPEN_FIGER_POS
        ring_finger_pos = CLOSE_FIGER_POS
        little_finger_pos = CLOSE_FIGER_POS
        thumb_rotation_pos = HALF_ROTATION_POS

    elif gesture == Gesture.THUMBS_UP:
        thumb_finger_pos = OPEN_FIGER_POS
        index_finger_pos = CLOSE_FIGER_POS
        middle_finger_pos = CLOSE_FIGER_POS
        ring_finger_pos = CLOSE_FIGER_POS
        little_finger_pos = CLOSE_FIGER_POS
        thumb_rotation_pos = SIDE_ROTATION_POS

    else:
        logger.warning("Unknown gesture")
        pass

    return thumb_finger_pos, index_finger_pos, middle_finger_pos, ring_finger_pos, little_finger_pos, thumb_rotation_pos
