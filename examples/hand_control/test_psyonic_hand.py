print("Loading hand control test...")
# from control.interface_control import InterfaceControl
import time
from math import pi, sin
from pathlib import Path
from ah_wrapper import AHSerialClient

from emager_tools.control.gesture_decoder import decode_gesture
from emager_tools.utils.utils import print_packet
from emager_tools.config.load_config import load_config
from emager_tools.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
parser = create_parser(
    description="Test Psyonic prosthetic hand control",
    default_config=str(DEFAULT_CONFIG)
)
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging (after loading config so it can use config defaults)
setup_logging(args, cfg, script_name="test_psyonic_hand")

# Save config if requested
save_config_if_requested(args, cfg, script_name="test_psyonic_hand")


def main():
    client = AHSerialClient(write_thread=False)
    """
    Since write_thread == False we will need to issue send_command after every 
    set_position command since the while loop is acting as our write thread.  One
    could alternatively generate a message using the api and use that as an argument
    for the send_command() method.  Using set_position allows you to update the 
    hands targets, which is useful for in the loop control.
    """
    
    try:
        print("\nTesting basic gestures...")
        pos = [30, 30, 30, 30, 30, -30]
        
        # gestures = ["Peace", "Hand_Close", "Hand_Open", "OK"]
        gestures = cfg.CLASSES  # Using gesture indices from config
        last_time = time.time()
        gesture_id = 0
        time_per_gesture = 1  # seconds
        while True:
            current_time = time.time()
            if current_time - last_time > time_per_gesture:  # Change gesture every second
                last_time = current_time
                gesture_id = (gesture_id + 1) % len(gestures)

            gesture = gestures[gesture_id]
            print(f"Sending gesture: {gesture}")
            # Get finger positions from our gesture decoder
            thumb_pos, index_pos, middle_pos, ring_pos, little_pos, thumb_rotation_pos = decode_gesture(gesture)
            # Scale positions from 0-1500 to 0-150 for Psyonic hand
            pos = [
                int(index_pos / 10),
                int(middle_pos / 10),
                int(ring_pos / 10),
                int(little_pos / 10),
                int(thumb_pos / 10),
                int(thumb_rotation_pos / 10)
            ]
            print(f"Setting positions to: {pos}")
            client.set_position(positions=pos, reply_mode=2)  # Update command
            client.send_command()  # Send command
            time.sleep(1 / client.rate_hz)
            
            
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
