import time
from pathlib import Path
import logging

from emagerlib.control.gesture_decoder import setup_gesture_decoder
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
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


args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging
setup_logging(args, cfg, script_name="test_teensy_hand")
logger = logging.getLogger(__name__)

# Save config if requested
save_config_if_requested(args, cfg, script_name="test_teensy_hand")

# Setup gesture decoder with config
setup_gesture_decoder(cfg)

logger.info("="*60)
logger.info("Interactive Commands - Psyonic Hand Control Test via Teensy ")
logger.info("="*60)


def main():
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


if __name__ == "__main__":
    main()
