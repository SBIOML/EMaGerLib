from emagerlib.control.constants import *
import emagerlib.utils.gestures_json as gj
from libemg.gui import GUI
import logging
from pathlib import Path
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# Module-level configuration - set via setup_gesture_decoder()
_MEDIA_PATH = None
_gestures_dict = None
_is_initialized = False

# FINGER POSITIONS PREDEFINED
NUTRAL_FINGER_POS = 25
OPEN_FINGER_POS = 0
CLOSE_FINGER_POS = 95
HALF_OPEN_FINGER_POS = 50
FRONT_ROTATION_POS = -100
HALF_ROTATION_POS = -50
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
    
    match gesture:
        case Gesture.NO_MOTION:
            thumb_finger_pos = NUTRAL_FINGER_POS
            index_finger_pos = NUTRAL_FINGER_POS
            middle_finger_pos = NUTRAL_FINGER_POS
            ring_finger_pos = NUTRAL_FINGER_POS
            little_finger_pos = NUTRAL_FINGER_POS
            thumb_rotation_pos = SIDE_ROTATION_POS

        case Gesture.HAND_CLOSE:
            thumb_finger_pos = HALF_OPEN_FINGER_POS
            index_finger_pos = CLOSE_FINGER_POS
            middle_finger_pos = CLOSE_FINGER_POS
            ring_finger_pos = CLOSE_FINGER_POS
            little_finger_pos = CLOSE_FINGER_POS
            thumb_rotation_pos = FRONT_ROTATION_POS

        case Gesture.HAND_OPEN:
            thumb_finger_pos = OPEN_FINGER_POS
            index_finger_pos = OPEN_FINGER_POS
            middle_finger_pos = OPEN_FINGER_POS
            ring_finger_pos = OPEN_FINGER_POS
            little_finger_pos = OPEN_FINGER_POS
            thumb_rotation_pos = SIDE_ROTATION_POS

        case Gesture.OK:
            thumb_finger_pos = 50
            index_finger_pos = 55
            middle_finger_pos = OPEN_FINGER_POS
            ring_finger_pos = OPEN_FINGER_POS
            little_finger_pos = OPEN_FINGER_POS
            thumb_rotation_pos = 75

        case Gesture.INDEX_EXTENSION:
            thumb_finger_pos = HALF_OPEN_FINGER_POS
            index_finger_pos = OPEN_FINGER_POS
            middle_finger_pos = CLOSE_FINGER_POS
            ring_finger_pos = CLOSE_FINGER_POS
            little_finger_pos = CLOSE_FINGER_POS
            thumb_rotation_pos = 75

        case Gesture.PEACE:
            thumb_finger_pos = 75
            index_finger_pos = OPEN_FINGER_POS
            middle_finger_pos = OPEN_FINGER_POS
            ring_finger_pos = CLOSE_FINGER_POS
            little_finger_pos = CLOSE_FINGER_POS
            thumb_rotation_pos = 75

        case Gesture.THUMBS_UP:
            thumb_finger_pos = OPEN_FINGER_POS
            index_finger_pos = CLOSE_FINGER_POS
            middle_finger_pos = CLOSE_FINGER_POS
            ring_finger_pos = CLOSE_FINGER_POS
            little_finger_pos = CLOSE_FINGER_POS
            thumb_rotation_pos = SIDE_ROTATION_POS
        
        case Gesture.ROCK_ON:
            thumb_finger_pos = NUTRAL_FINGER_POS
            index_finger_pos = OPEN_FINGER_POS
            middle_finger_pos = CLOSE_FINGER_POS
            ring_finger_pos = CLOSE_FINGER_POS
            little_finger_pos = OPEN_FINGER_POS
            thumb_rotation_pos = SIDE_ROTATION_POS

        case _:
            logger.warning("Unknown gesture")

    return thumb_finger_pos, index_finger_pos, middle_finger_pos, ring_finger_pos, little_finger_pos, thumb_rotation_pos

def decompose_mouvement(
    start_positions: Tuple[int, int, int, int, int, int],
    end_positions: Tuple[int, int, int, int, int, int],
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Identify situation and move out of the way accordingly making a list of positions to go through to avoid collisions.
    Args:
        start_positions: Starting finger positions (thumb, index, middle, ring, little, thumb_rot)
        end_positions: Target finger positions
    Returns:
        List of waypoints to move through to avoid collisions. Each waypoint is a tuple of finger positions.
    """
    # Either 3 or 2 or 1 (direct) waypoints depending on situation
    # 3 waypoints: Clear thumb first, then move fingers, then final thumb position
    # 2 waypoints: Move thumb to side first, then move fingers and (thumb rotation stay at the side)
    # 1 waypoint: Direct move if thumb is staying at the side or fingers ar not moving at all
    
    waypoints = []
    finger_mouvement = any(start != end for start, end in zip(start_positions[:4], end_positions[:4]))
    fingers_end_down = any(end > 50 for end in end_positions[:4])  # Check if any finger is closed in the target gesture
    fingers_start_down = any(start > 50 for start in start_positions[:4])  # Check if any finger is closed in the starting gesture
    thumb_start_side = abs(start_positions[5]) <= 10
    thumb_end_side = abs(end_positions[5]) <= 10
    
    finger_go_down = fingers_end_down and not fingers_start_down
    finger_go_up = not fingers_end_down and fingers_start_down
    thumb_go_side = thumb_end_side and not thumb_start_side
    thumb_go_front = not thumb_end_side and thumb_start_side
    
    # Situation 1 Direct move if thumb is staying at the side 
    # OR fingers are not moving at all (thumb can move in this case since it won't cause collision)
    if (thumb_start_side and thumb_end_side) or not finger_mouvement:
        waypoints.append(end_positions)
        logger.info("Direct move: Thumb stays at the side")
    # Situation 2 Move thumb to side first, then move fingers and (thumb rotation stay at the side)
    elif (thumb_go_side and finger_mouvement):
        # Move fhumb first
        waypoints.append((start_positions[0], start_positions[1], start_positions[2], start_positions[3], start_positions[4], end_positions[5]))  
        waypoints.append(end_positions)  # Then move fingers and thumb rotation to final position
        logger.info("Two-step move: Move thumb to side first, then move fingers")
    elif (thumb_go_front and finger_mouvement):
        # Move finger first
        waypoints.append((end_positions[0], end_positions[1], end_positions[2], end_positions[3], end_positions[4], start_positions[5]))  
        waypoints.append(end_positions)  # Then move thumb rotation to final position
        logger.info("Two-step move: Move fingers first, then move thumb rotation to front")
    # Situation 3 Move thumb to side first, then move fingers and thumb rotation (if thumb is going from front to side and fingers are moving, we want to move the thumb out of the way first before moving the fingers to avoid collision)
    elif ((not thumb_start_side and not thumb_end_side) and finger_mouvement):
        # Move thumb to the side first
        waypoints.append((OPEN_FINGER_POS, start_positions[1], start_positions[2], start_positions[3], start_positions[4], SIDE_ROTATION_POS))
        waypoints.append((OPEN_FINGER_POS, end_positions[1], end_positions[2], end_positions[3], end_positions[4], SIDE_ROTATION_POS))
        waypoints.append(end_positions)  # Then move fingers and thumb rotation to final position
        logger.info("Two-step move: Move thumb to side first, then move fingers and thumb rotation")
    
    return waypoints




def decode_gesture_waypoints(
    current_gesture: int,
    last_gesture: Optional[int] = None,
    last_positions: Optional[Tuple[int, int, int, int, int, int]] = None,
    collision_enabled: bool = True
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Decode a gesture to critical waypoints that avoid thumb-to-fingers collisions.
    Returns only key positions needed to move from last gesture to current gesture
    while preventing thumb from colliding with other fingers. Incremental movements 
    between waypoints are handled by the caller.
    Args:
        current_gesture: Gesture ID (int) or name (str) to move to
        last_gesture: Previous gesture ID (int) or name (str) (optional)
        last_positions: Previous finger positions as tuple (optional, overrides last_gesture)
        thumb_collision_threshold: Thumb flexion threshold for collision detection (default: 250)
        collision_enabled: Enable collision avoidance (default: True)
    Returns:
        List of critical waypoint tuples [(thumb, index, middle, ring, little, thumb_rot), ...]
        Typically 1-2 waypoints. Final waypoint reaches the target gesture.
    """
    _ensure_initialized()
    
    # Get current gesture positions
    current_positions = decode_gesture(current_gesture)
    
    # Determine starting positions
    if last_positions is not None:
        start_positions = last_positions
        logger.debug(f"Using provided last positions: {start_positions}")
    elif last_gesture is not None:
        start_positions = decode_gesture(last_gesture)
        logger.debug(f"Using last gesture {last_gesture}: {start_positions}")
    else:
        start_positions = (NUTRAL_FINGER_POS, NUTRAL_FINGER_POS, NUTRAL_FINGER_POS, 
                          NUTRAL_FINGER_POS, NUTRAL_FINGER_POS, SIDE_ROTATION_POS)
        logger.debug(f"Using neutral positions: {start_positions}")
    
    # Identify collision waypoints
    if collision_enabled:
        waypoints = decompose_mouvement(
            start_positions, 
            current_positions
        )
    else:
        waypoints = [current_positions]
    
    logger.info(f"Generated {len(waypoints)} waypoints for gesture transition")
    logger.debug(f"Start: {start_positions}, Waypoints: {waypoints}")
    
    return waypoints

