"""
Real-time EMG control using Psyonic Hand via Teensy Controller.

Simple 2-thread architecture with buffers:
- Thread 1: Predictor - reads EMG and makes predictions at PREDICTOR_DELAY rate
- Thread 2: Controller - reads predictions and sends to Teensy at CONTROLLER_POLL_RATE

Buffers:
- prediction_buffer: Queue of predictions from LibEMG
- send_buffer: Tracks last sent gesture to avoid duplicates
"""

import threading
import time
from queue import Queue, Empty
from pathlib import Path
import logging
import serial
import serial.tools.list_ports

from libemg.environments.controllers import ClassifierController
from examples.realtime.realtime_prediction import predicator
import emagerlib.utils.gestures_json as gjutils
from emagerlib.control.interface_control import InterfaceControl
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
parser = create_parser(
    description="Real-time EMG control via Psyonic Teensy Controller",
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
setup_logging(args, cfg, script_name="realtime_control_teensy")
logger = logging.getLogger(__name__)

# Save config if requested
save_config_if_requested(args, cfg, script_name="realtime_control_teensy")

# Shared resources
stop_event = threading.Event()
prediction_buffer = Queue(maxsize=10)  # Buffer for predictions from LibEMG


# PREDICTOR
def run_predictor_thread(ctrl):
    """
    Thread 1: Runs predictor and reads from LibEMG shared memory.
    Speed controlled by: cfg.PREDICTOR_DELAY
    
    Args:
        ctrl: Shared ClassifierController instance (avoid duplicate socket binding)
    """
    try:
        logger.info("Starting predictor worker...")
        
        # Run predictor (sets up LibEMG classifier)
        predicator(
            use_gui=cfg.USE_GUI,
            ctrl_shared=ctrl,  # Pass shared controller
            conn=None,  # No Pipe needed!
            delay=cfg.PREDICTOR_DELAY,
            timeout_delay=cfg.PREDICTOR_TIMEOUT_DELAY
        )
        
    except Exception as e:
        logger.error(f"Error in predictor worker: {e}", exc_info=True)
    finally:
        stop_event.set()
        logger.info("Predictor worker stopped")


# CONTROLLER
def run_controller_thread(ctrl):
    """
    Thread 2: Reads predictions from LibEMG shared memory and sends to Teensy.
    Speed controlled by: cfg.CONTROLLER_POLL_RATE
    
    Args:
        ctrl: Shared ClassifierController instance (avoid duplicate socket binding)
    """
    try:
        logger.info("Starting controller worker...")
        
        # Wait for predictor to initialize
        time.sleep(3)
        
        # Connect to Teensy controller
        logger.info(f"Connecting to Teensy on port: {args.port or 'auto-detect'}")
        comm_controller = InterfaceControl(
            hand_type="psyonic_teensy",
            cfg=cfg,
            port=args.port,
            baudrate=args.baudrate
        )
        comm_controller.connect()
        logger.info("Teensy controller connected and ready")
        
        # Load gesture mappings
        gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
        images = gjutils.get_images_list(cfg.MEDIA_PATH)
        
        logger.info("Connected to LibEMG shared memory")
        
        # Send buffer: track last sent gesture
        last_sent_gesture = None
        
        logger.info("Controller ready, waiting for predictions...")
        logger.info(f"Controller poll rate: {cfg.CONTROLLER_POLL_RATE}s")
        
        while not stop_event.is_set():
            # Read prediction from LibEMG shared memory
            predictions = ctrl.get_data(['predictions'])
            
            if predictions is None:
                time.sleep(cfg.CONTROLLER_POLL_RATE)
                continue
            
            # Convert prediction index to gesture name
            try:
                pred_idx = int(predictions[0])
                
                if pred_idx not in range(cfg.NUM_CLASSES):
                    pred_idx = 0
                
                gesture = gjutils.get_label_from_index(pred_idx, images, gestures_dict)
                
                if gesture is None:
                    logger.error(f"Failed to get gesture for index {pred_idx}")
                    time.sleep(cfg.CONTROLLER_POLL_RATE)
                    continue
                
            except Exception as e:
                logger.error(f"Error processing prediction: {e}")
                time.sleep(cfg.CONTROLLER_POLL_RATE)
                continue
            
            # Only send if gesture changed
            if gesture != last_sent_gesture:
                try:
                    comm_controller.send_gesture(gesture)
                    last_sent_gesture = gesture
                    logger.info(f"Sent gesture [{gesture}] to Teensy")
                except Exception as e:
                    logger.error(f"Error sending gesture: {e}")
            
            # Control polling rate
            time.sleep(cfg.CONTROLLER_POLL_RATE)
        
        logger.info("Controller worker stopped")
        
    except Exception as e:
        logger.error(f"Error in controller worker: {e}", exc_info=True)
    finally:
        if 'comm_controller' in locals():
            try:
                logger.info("Disconnecting from Teensy...")
                comm_controller.disconnect()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        stop_event.set()


def main():
    """Main entry point."""
    try:
        logger.info("="*60)
        logger.info("Real-time EMG Control via Psyonic Teensy Controller")
        logger.info("="*60)
        logger.info(f"Configuration: {args.config}")
        logger.info(f"GUI enabled: {cfg.USE_GUI}")
        logger.info(f"Predictor speed: {cfg.PREDICTOR_DELAY}s between predictions")
        logger.info(f"Controller speed: {cfg.CONTROLLER_POLL_RATE}s between polls")
        logger.info(f"Teensy port: {args.port or 'auto-detect'}")
        logger.info("="*60)
        
        ports = serial.tools.list_ports.comports()
        logger.info("Available serial ports:")
        for i, port in enumerate(ports):
            logger.info(f"  [{i}] {port.device}: {port.description}")
        
        
        # Create shared ClassifierController (only once to avoid socket conflicts on Windows)
        logger.info("Connecting to LibEMG shared memory...")
        ctrl = ClassifierController('predictions', cfg.NUM_CLASSES)
        logger.info("✓ LibEMG shared memory connected")

        # Start controller thread
        controller_thread = threading.Thread(
            target=run_controller_thread,
            args=(ctrl,),
            daemon=True,
            name="Controller"
        )
        controller_thread.start()
        logger.info("✓ Controller thread started")

        # Start predictor (GUI mode or thread mode)
        if cfg.USE_GUI:
            logger.info("GUI enabled: running predictor in main thread.")
            try:
                # Run predictor in main thread (GUI must be here)
                run_predictor_thread(ctrl)
            except Exception as e:
                logger.error(f"Fatal error in predictor: {e}", exc_info=True)
        else:
            predictor_thread = threading.Thread(
                target=run_predictor_thread,
                args=(ctrl,),
                daemon=True,
                name="Predictor"
            )
            predictor_thread.start()
            logger.info("✓ Predictor thread started")

            logger.info("")
            logger.info("System running. Press Ctrl+C to stop.")
            logger.info("")

            # Keep main thread alive
            while not stop_event.is_set():
                time.sleep(0.1)

        # Shutdown handling
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Keyboard interrupt received - shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Stopping all workers...")
        stop_event.set()

        # Wait for threads to finish
        if 'predictor_thread' in locals() and predictor_thread.is_alive():
            predictor_thread.join(timeout=3)
            if predictor_thread.is_alive():
                logger.warning("⚠ Predictor thread did not stop cleanly")
            else:
                logger.info("✓ Predictor thread stopped")

        if controller_thread.is_alive():
            controller_thread.join(timeout=3)
            if controller_thread.is_alive():
                logger.warning("⚠ Controller thread did not stop cleanly")
            else:
                logger.info("✓ Controller thread stopped")

        logger.info("="*60)
        logger.info("Application exited")
        logger.info("="*60)


if __name__ == "__main__":
    main()
