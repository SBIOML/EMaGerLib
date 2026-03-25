"""
Test Psyonic Hand Control via Teensy Controller

This script demonstrates basic usage of the Teensy-based controller for the Psyonic hand.
It can run in four modes:

1. Gesture Test Mode (default): Cycles through gestures from configuration
2. Individual Finger Test: Tests each finger independently  
3. Smooth Gesture Waypoints: Tests collision-aware waypoint control
4. Interactive Mode: Manual command-line UART testing

Hardware Setup:
- Teensy connected to computer via USB
- Teensy connected to Psyonic hand
- Teensy firmware running hand control code

Usage:
    # Standard gesture test
    python test_teensy_hand.py [--config CONFIG_PATH] [--port COM3]
    
    # Smooth gesture waypoint control with collision avoidance
    python test_teensy_hand.py --test-waypoints [--port COM3]
    
    # Individual finger test
    python test_teensy_hand.py --test-fingers [--port COM3]
    
    # Interactive UART command mode
    python test_teensy_hand.py --interactive [--port COM3]
"""

import time
from pathlib import Path
import logging

from emagerlib.control.psyonic_teensy_control import PsyonicTeensyControl
from emagerlib.control.gesture_decoder import (
    decode_gesture,
    setup_gesture_decoder
)
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Test Psyonic hand control via Teensy controller",
        default_config=str(DEFAULT_CONFIG)
    )
    parser.add_argument(
        '--port',
        type=str,
        default=None,
        help='Serial port for Teensy (e.g., COM3, /dev/ttyACM0). Auto-detects if not specified.'
    )
    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Serial baudrate for Teensy communication (default: 115200)'
    )
    parser.add_argument(
        '--gesture-duration',
        type=float,
        default=2.0,
        help='Duration to hold each gesture in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive command-line mode for manual UART testing'
    )
    parser.add_argument(
        '--test-fingers',
        action='store_true',
        help='Test individual finger control instead of gestures'
    )
    parser.add_argument(
        '--test-controller2',
        action='store_true',
        help='Test the second Teensy controller (if available) instead of the main hand controller'
    )
    parser.add_argument(
        '--test-delays',
        action='store_true',
        help='Test communication delays and responsiveness'
    )
    parser.add_argument(
        '--test-waypoints',
        action='store_true',
        help='Test smooth gesture control with collision-aware waypoints'
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="test_teensy_hand")
    save_config_if_requested(args, cfg, script_name="test_teensy_hand")
    setup_gesture_decoder(cfg)
    logger.info("=" * 60)
    logger.info("Psyonic Hand Control via Teensy - Test Script")
    logger.info("=" * 60)
    return args, cfg


def run_main_gesture_test(args, cfg):
    """Main test function."""
    logger.info(f"Port: {args.port or 'auto-detect'}")
    logger.info(f"Baudrate: {args.baudrate}")
    logger.info(f"Gesture duration: {args.gesture_duration}s")
    logger.info(f"Available gestures: {cfg.CLASSES}")
    logger.info("")
    
    # Create and connect to Teensy controller
    try:
        logger.info("Connecting to Teensy controller...")
        controller = PsyonicTeensyControl(port=args.port, baudrate=args.baudrate)
        controller.connect()
        logger.info("✓ Connected successfully!")
        logger.info("")
        
        # Give some time for initialization
        time.sleep(1)
        
        # Start the hand thread on the Teensy
        controller.teensy._send_command('t 1')  # Start hand thread
        time.sleep(0.5)
        
        # Test basic gestures
        logger.info("Starting gesture test sequence...")
        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        gestures = cfg.CLASSES
        gesture_id = 0
        
        while True:
            gesture = gestures[gesture_id]
            logger.info(f"Sending gesture: [{gesture}]")
            
            try:
                # Send gesture to Teensy
                controller.send_gesture(gesture)
                
                # Get the actual positions for display
                positions = decode_gesture(gesture)
                
                logger.info(f"  Positions: {positions}")
                logger.info(f"  Holding for {args.gesture_duration}s...")
                
            except Exception as e:
                logger.error(f"Error sending gesture: {e}")
            
            # Wait before next gesture
            time.sleep(args.gesture_duration)
            
            # Move to next gesture
            gesture_id = (gesture_id + 1) % len(gestures)
            logger.info("")
            
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        if 'controller' in locals():
            logger.info("")
            logger.info("Disconnecting from Teensy...")
            try:
                controller.disconnect()
                logger.info("✓ Disconnected")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        logger.info("")
        logger.info("="*60)
        logger.info("Test completed")
        logger.info("="*60)


def test_individual_fingers(args):
    """Test individual finger control (alternative test mode)."""
    logger.info("Testing individual finger control...")
    
    controller = PsyonicTeensyControl(port=args.port, baudrate=args.baudrate)
    
    try:
        controller.connect()
        logger.info("Connected successfully!")
        
        # Test each finger individually
        fingers = [
            ("Thumb Flex", 0),
            ("Index", 1),
            ("Middle", 2),
            ("Ring", 3),
            ("Pinky", 4),
            ("Thumb Rotation", 5)
        ]
        
        for finger_name, finger_id in fingers:
            logger.info(f"Testing {finger_name} (finger {finger_id})")
            
            # Move to 100 (or 90 for thumb rotation)
            max_pos = 100
            logger.info(f"  Moving to {max_pos}...")
            controller.send_finger_position(finger_id, max_pos)
            time.sleep(1.5)
            
            # Move back to 0
            logger.info(f"  Moving to 0...")
            controller.send_finger_position(finger_id, 0)
            time.sleep(1.5)
            
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error during finger test: {e}")
    finally:
        controller.disconnect()


def interactive_command_mode(args):
    """Interactive command-line mode for manual UART testing."""
    from emagerlib.control.psyonic_teensy_control import PsyonicTeensyController
    
    logger.info("="*60)
    logger.info("Interactive UART Command Mode")
    logger.info("="*60)
    logger.info("")
    logger.info("This mode allows you to send raw commands to the Teensy")
    logger.info("and see responses in real-time.")
    logger.info("")
    
    try:
        # Create low-level controller with visible responses
        logger.info(f"Connecting to Teensy on {args.port or 'auto-detect'}...")
        teensy = PsyonicTeensyController(
            port=args.port or PsyonicTeensyController.find_teensy_port(),
            baudrate=args.baudrate,
            auto_read=True,
            response_callback=lambda line: print(f"[Teensy] {line}")
        )
        
        logger.info("✓ Connected successfully!\n")
        
        # Start the hand thread
        logger.info("Starting hand thread...")
        teensy._send_command('t 1')
        time.sleep(0.5)
        
        # Get initial status
        logger.info("Getting initial status...")
        teensy._send_command('s')
        time.sleep(0.5)
        logger.info("")
        
        # Interactive loop
        while True:
            try:
                # Get command from user
                cmd = input("Teensy> ").strip()
                
                if not cmd:
                    continue
                
                # Handle special commands
                if cmd.lower() in ['quit', 'exit']:
                    logger.info("Exiting interactive mode...")
                    break
                
                if cmd.lower() == 'help':
                    teensy._send_command('m')
                    continue
                
                # Send command to Teensy
                teensy._send_command(cmd)
                
            except EOFError:
                logger.info("")
                logger.info("EOF received, exiting...")
                break
            except KeyboardInterrupt:
                logger.info("")
                logger.info("Interrupted by user...")
                break
        
        time.sleep(0.5)
        
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}", exc_info=True)
    finally:
        if 'teensy' in locals():
            teensy.close()
        
        logger.info("")
        logger.info("="*60)
        logger.info("Interactive mode exited")
        logger.info("="*60)


def main(argv=None):
    args, cfg = setup_runtime(argv)

    if args.interactive:
        interactive_command_mode(args)
    elif args.test_fingers:
        test_individual_fingers(args)
    else:
        run_main_gesture_test(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
