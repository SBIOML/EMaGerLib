import time
from math import pi, sin
from pathlib import Path
import logging

from emagerlib.control.gesture_decoder import decode_gesture, setup_gesture_decoder
from emagerlib.utils.utils import print_packet
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Test Psyonic prosthetic hand control",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="test_psyonic_hand")
    save_config_if_requested(args, cfg, script_name="test_psyonic_hand")
    setup_gesture_decoder(cfg)
    logger.info("Loading hand control test...")
    return args, cfg


def main(argv=None):
    from ah_wrapper import AHSerialClient

    _, cfg = setup_runtime(argv)
    client = AHSerialClient(write_thread=False)
    """
    Since write_thread == False we will need to issue send_command after every 
    set_position command since the while loop is acting as our write thread.  One
    could alternatively generate a message using the api and use that as an argument
    for the send_command() method.  Using set_position allows you to update the 
    hands targets, which is useful for in the loop control.
    """
    
    try:
        logger.info("Testing basic gestures...")
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
            logger.info(f"Sending gesture: {gesture}")
            # Get finger positions from our gesture decoder
            positions = decode_gesture(gesture)
            
            logger.debug(f"Setting positions to: {positions}")
            client.set_position(positions=positions, reply_mode=2)  # Update command
            client.send_command()  # Send command
            time.sleep(1 / client.rate_hz)
            
            
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
