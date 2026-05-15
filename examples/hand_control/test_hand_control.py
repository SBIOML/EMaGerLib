import time
from pathlib import Path
import logging

from emagerlib.control.interface_control import InterfaceControl
from emagerlib.utils.utils import print_packet
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Test prosthetic hand control interface",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="test_hand_control")
    save_config_if_requested(args, cfg, script_name="test_hand_control")
    logger.info("Loading hand control test...")
    return args, cfg


def test_hand(hand_type, cfg, **kwargs):
    logger.info(f"=== Testing {hand_type.capitalize()} Hand ===")
    try:
        # Initialize hand
        hand = InterfaceControl(hand_type=hand_type, **kwargs)
        
        # Connect to the device
        logger.info(f"Connecting to {hand_type} hand...")
        hand.connect()
        
        # Test basic functionality
        logger.info("Testing basic gestures...")
        # gestures = ["Peace", "Hand_Close", "Hand_Open", "OK"]
        gestures = cfg.CLASSES
        for gesture in gestures:
            logger.info(f"Sending gesture: {gesture}")
            hand.send_gesture(gesture)
            time.sleep(1)  # Wait between gestures
            # logger.debug("Reading data...")
            # packet = hand.read_data()
            # print_packet(packet, stuffed=True)
            # time.sleep(0.5) 
            
            
        # Test individual finger control
        # logger.info("Testing individual finger control...")
        # for finger in range(6):
        #     logger.info(f"Moving finger {finger} to position 50")
        #     hand.send_finger_position(finger, 100)
        #     time.sleep(0.2)
        
            
        # Test hand-specific features
        if hand_type == "zeus":
            logger.info("Testing telemetry...")
            hand.start_telemetry()
            time.sleep(2)
            hand.stop_telemetry()
            
        elif hand_type == "smart":
            # Test direct gesture interface
            logger.info("Testing direct gesture interface...")
            for gesture_value in range(5):  # Test gestures 0-4
                logger.info(f"Sending direct gesture value: {gesture_value}")
                hand.send_gesture(gesture_value, direct=True)
                time.sleep(0.5)
                
            # Test LED controls
            logger.info("Testing LED controls...")
            hand.hand.toggle_led_rpi()
            time.sleep(1)
            hand.hand.blink_led_rpi()
            
        # Cleanup
        logger.info("Disconnecting...")
        hand.disconnect()
        
    except Exception as e:
        logger.error(f"Error testing {hand_type} hand: {e}")
        raise e

def main():
    _, cfg = setup_runtime()
    logger.info("Starting hand control tests...")
    
    try:
        test_hand("psyonic", cfg, stuffing=True)
        logger.info("All tests completed!") 
    except Exception as e:
        logger.error(f"Tests Failed! : {e}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
        
    
    