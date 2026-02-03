print("Loading hand control test...")
import time
from pathlib import Path

from emager_tools.control.interface_control import InterfaceControl
from emager_tools.utils.utils import print_packet
from emager_tools.config.load_config import load_config
from emager_tools.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
parser = create_parser(
    description="Test prosthetic hand control interface",
    default_config=str(DEFAULT_CONFIG)
)
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging (after loading config so it can use config defaults)
setup_logging(args, cfg, script_name="test_hand_control")

# Save config if requested
save_config_if_requested(args, cfg, script_name="test_hand_control")

def test_hand(hand_type, **kwargs):
    print(f"\n=== Testing {hand_type.capitalize()} Hand ===")
    try:
        # Initialize hand
        hand = InterfaceControl(hand_type=hand_type, **kwargs)
        
        # Connect to the device
        print(f"Connecting to {hand_type} hand...")
        hand.connect()
        
        # Test basic functionality
        print("\nTesting basic gestures...")
        # gestures = ["Peace", "Hand_Close", "Hand_Open", "OK"]
        gestures = cfg.CLASSES  # Using gesture indices from config
        for gesture in gestures:
            print(f"Sending gesture: {gesture}")
            hand.send_gesture(gesture)
            time.sleep(1)  # Wait between gestures
            # print("Reading data...")
            # packet = hand.read_data()
            # print_packet(packet, stuffed=True)
            # time.sleep(0.5) 
            
            
        # Test individual finger control
        # print("\nTesting individual finger control...")
        # for finger in range(6):
        #     print(f"Moving finger {finger} to position 50")
        #     hand.send_finger_position(finger, 100)
        #     time.sleep(0.2)
        
            
        # Test hand-specific features
        if hand_type == "zeus":
            print("\nTesting telemetry...")
            hand.start_telemetry()
            time.sleep(2)
            hand.stop_telemetry()
            
        elif hand_type == "smart":
            # Test direct gesture interface
            print("\nTesting direct gesture interface...")
            for gesture_value in range(5):  # Test gestures 0-4
                print(f"Sending direct gesture value: {gesture_value}")
                hand.send_gesture(gesture_value, direct=True)
                time.sleep(0.5)
                
            # Test LED controls
            print("\nTesting LED controls...")
            hand.hand.toggle_led_rpi()
            time.sleep(1)
            hand.hand.blink_led_rpi()
            
        # Cleanup
        print("\nDisconnecting...")
        hand.disconnect()
        
    except Exception as e:
        print(f"Error testing {hand_type} hand: {e}")
        raise e

def main():
    print("\n Starting hand control tests...")
    
    try:
        test_hand("psyonic", stuffing=True)
        print("\nAll tests completed!") 
    except Exception as e:
        print(f"Tests Failed! : {e}")

if __name__ == "__main__":
    main()
        
    
    