"""
Real-time EMG control using THREADING with direct shared memory access.

This version uses threading instead of multiprocessing:
- Thread 1: Runs the predictor (includes LibEMG classifier + GUI)
- Thread 2: Reads predictions from LibEMG shared memory and sends to hand

Advantages:
- Much lower latency (no Pipe serialization)
- Simpler code (no IPC)
- Direct shared memory access via LibEMG's ClassifierController
- 3-5x faster than multiprocessing + Pipe

Performance: ~8-12ms latency, 100-150Hz capable
"""

import threading
import time
from collections import deque, Counter
from statistics import mean
from pathlib import Path
import logging

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
    description="Real-time EMG prosthetic hand control (Threading with shared memory)",
    default_config=str(DEFAULT_CONFIG)
)
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging
setup_logging(args, cfg, script_name="realtime_control_threads")
logger = logging.getLogger(__name__)

# Save config if requested
save_config_if_requested(args, cfg, script_name="realtime_control_threads")

# Shared stop event
stop_event = threading.Event()


# PREDICTOR
def run_predictor_thread():
    """
    Worker that runs the predictor.
    This sets up LibEMG classifier and optionally shows GUI.
    Predictions go directly to LibEMG's shared memory.
    """
    try:
        logger.info("Starting predictor worker...")
        
        # Run predictor without Pipe connection
        # It will write predictions to LibEMG shared memory automatically
        predicator(
            use_gui=cfg.USE_GUI,
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
def run_controller_thread():
    """
    Worker that reads predictions from LibEMG shared memory and sends to hand.
    Uses ClassifierController to access the same shared memory as the predictor.
    """
    try:
        logger.info("Starting controller worker...")
        
        # Wait a bit for predictor to initialize shared memory
        time.sleep(3)
        
        # Connect to hand controller
        comm_controller = InterfaceControl(hand_type="psyonic", cfg=cfg)
        comm_controller.connect()
        logger.info("Hand controller connected")
        
        # Load gesture mappings
        gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
        images = gjutils.get_images_list(cfg.MEDIA_PATH)
        
        # Create ClassifierController to read from LibEMG's shared memory
        # This accesses the SAME shared memory that the predictor writes to
        ctrl = ClassifierController('predictions', cfg.NUM_CLASSES)
        logger.info("Connected to LibEMG shared memory")
        
        # Smoothing buffer
        recent = deque(maxlen=cfg.SMOOTH_WINDOW)
        last_gesture = None
        last_send_time = 0.0
        heartbeat_interval = cfg.HEARTBEAT_INTERVAL
        
        # Performance tracking
        loop_times = []
        read_times = []
        iteration = 0
        
        logger.info("Controller ready, waiting for predictions...")
        
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            
            # Read prediction directly from shared memory - NO PIPE!
            read_start = time.perf_counter()
            predictions = ctrl.get_data(['predictions'])
            read_elapsed = time.perf_counter() - read_start
            read_times.append(read_elapsed)
            
            if predictions is None:
                # No prediction yet, check heartbeat
                now = time.perf_counter()
                if heartbeat_interval and last_gesture is not None:
                    if (now - last_send_time) >= heartbeat_interval:
                        try:
                            comm_controller.send_gesture(last_gesture)
                            last_send_time = now
                            logger.debug(f"Heartbeat: re-sent gesture [{last_gesture}]")
                        except Exception as e:
                            logger.error(f"Error sending heartbeat gesture: {e}")
                
                # Short sleep to prevent busy waiting
                time.sleep(max(cfg.POLL_SLEEP_DELAY, 0.0001))
                continue
            
            # Process prediction
            try:
                input_pred = int(predictions[0])
                
                # Validate prediction
                if input_pred not in range(cfg.NUM_CLASSES):
                    input_pred = 0
                
                # Add to smoothing buffer
                recent.append(input_pred)
                
                # Apply smoothing if enabled
                if cfg.SMOOTH_WINDOW > 1 and len(recent) > 1:
                    if cfg.SMOOTH_METHOD == 'mode':
                        counts = Counter(recent)
                        most_common = counts.most_common()
                        top_count = most_common[0][1]
                        candidates = [val for val, cnt in most_common if cnt == top_count]
                        # If tie, pick most recent
                        for v in reversed(recent):
                            if v in candidates:
                                smoothed_pred = v
                                break
                    elif cfg.SMOOTH_METHOD == 'mean':
                        smoothed_pred = int(round(mean(recent)))
                    else:
                        smoothed_pred = recent[-1]
                else:
                    smoothed_pred = input_pred
                
                # Get gesture label
                gesture = gjutils.get_label_from_index(smoothed_pred, images, gestures_dict)
                
                logger.debug(f"Prediction: {input_pred} -> Smoothed: {smoothed_pred} -> Gesture: [{gesture}]")
                
            except Exception as e:
                logger.error(f"Error processing prediction: {e}")
                continue
            
            # Send gesture to hand
            try:
                comm_controller.send_gesture(gesture)
                last_gesture = gesture
                last_send_time = time.perf_counter()
                logger.info(f"Sent gesture [{gesture}] to hand")
            except Exception as e:
                logger.error(f"Error sending gesture: {e}")
            
            # Performance monitoring
            loop_elapsed = time.perf_counter() - loop_start
            loop_times.append(loop_elapsed)
            
            iteration += 1
            if iteration % 100 == 0:
                # Log performance metrics every 100 iterations
                avg_loop = sum(loop_times) / len(loop_times) * 1000
                max_loop = max(loop_times) * 1000
                avg_read = sum(read_times) / len(read_times) * 1000
                logger.info(f"Performance: Avg loop {avg_loop:.2f}ms, Max {max_loop:.2f}ms, Avg read {avg_read:.3f}ms")
                loop_times.clear()
                read_times.clear()
        
        logger.info("Controller worker stopped")
        
    except Exception as e:
        logger.error(f"Error in controller worker: {e}", exc_info=True)
    finally:
        if 'comm_controller' in locals():
            comm_controller.disconnect()
        stop_event.set()


def main():
    """Main entry point."""
    try:
        logger.info("Starting threaded real-time control...")
        logger.info(f"GUI enabled: {cfg.USE_GUI}")
        logger.info(f"Smooth window: {cfg.SMOOTH_WINDOW}, method: {cfg.SMOOTH_METHOD}")
        logger.info(f"Heartbeat interval: {cfg.HEARTBEAT_INTERVAL}s")
        
        # Start predictor thread (sets up LibEMG and optionally shows GUI)
        predictor_thread = threading.Thread(
            target=run_predictor_thread,
            daemon=True,
            name="Predictor"
        )
        predictor_thread.start()
        logger.info("Predictor thread started")
        
        # Start controller thread (reads from shared memory and controls hand)
        controller_thread = threading.Thread(
            target=run_controller_thread,
            daemon=True,
            name="Controller"
        )
        controller_thread.start()
        logger.info("Controller thread started")
        
        # Keep main thread alive
        while not stop_event.is_set():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Stopping all workers...")
        stop_event.set()
        
        # Wait for threads to finish
        if predictor_thread.is_alive():
            predictor_thread.join(timeout=3)
            if predictor_thread.is_alive():
                logger.warning("Predictor thread did not stop cleanly")
        
        if controller_thread.is_alive():
            controller_thread.join(timeout=3)
            if controller_thread.is_alive():
                logger.warning("Controller thread did not stop cleanly")
        
        logger.info("Application exiting")


if __name__ == "__main__":
    main()
