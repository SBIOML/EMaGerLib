from control.constants import *
import utils.gestures_json as gj
from config_emager import *
from libemg.gui import GUI

def log(message, mode=Logger.INFO):
    print(message)

# FINGER POSITIONS PREDEFINED
NUTRAL_FINGER_POS = 250
OPEN_FIGER_POS = 0
CLOSE_FIGER_POS = 1000
HALF_OPEN_FIGER_POS = 500
FRONT_ROTATION_POS = 0
HALF_ROTATION_POS = -500
SIDE_ROTATION_POS = -1000

try:
    gestures_dict = gj.get_gestures_dict(MEDIA_PATH)
except FileNotFoundError:
    GUI.download_gestures(MEDIA_PATH)
    gestures_dict = gj.get_gestures_dict(MEDIA_PATH)

def decode_gesture(gesture):
    if isinstance(gesture, int):
        gestures_name = gestures_dict[str(gesture)]
    if isinstance(gesture, str):
        gestures_name = gesture
        gesture = int(list(gestures_dict.keys())[list(gestures_dict.values()).index(gestures_name)])
    print(f"Gesture: {gestures_name} ({gesture})")

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
        log("Unknown gesture", mode=Logger.WARNING)
        pass

    return thumb_finger_pos, index_finger_pos, middle_finger_pos, ring_finger_pos, little_finger_pos, thumb_rotation_pos
