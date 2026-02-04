from multiprocessing.connection import Connection
from multiprocessing import Lock, Process, Pipe
import time
from collections import deque, Counter
from statistics import mean
from pathlib import Path
import logging

from examples.realtime.realtime_prediction import predicator
import emager_tools.utils.utils as eutils
import emager_tools.utils.gestures_json as gjutils
from emager_tools.control.interface_control import InterfaceControl
from emager_tools.config.load_config import load_config
from emager_tools.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Parse arguments
parser = create_parser(
    description="Real-time EMG prosthetic hand control",
    default_config=str(DEFAULT_CONFIG)
)
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Setup logging (after loading config so it can use config defaults)
setup_logging(args, cfg, script_name="realtime_control")

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)

# Save config if requested (uses logging internally)
save_config_if_requested(args, cfg, script_name="realtime_control")

# PREDICTOR
def run_predicator_process(conn: Connection=None):
    predicator(use_gui=cfg.USE_GUI, conn=conn, delay=cfg.PREDICTOR_DELAY, timeout_delay=cfg.PREDICTOR_TIMEOUT_DELAY)


# COMMUNICATOR
def run_controller_process(conn: Connection=None):
    try:
        
        comm_controller = InterfaceControl(hand_type="psyonic")
        comm_controller.connect()
        
        gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
        images = gjutils.get_images_list(cfg.MEDIA_PATH)
        
        # Main loop to read input from stdin
        logger.info("Communicator waiting for data...")
        recent = deque(maxlen=cfg.SMOOTH_WINDOW)
        last_gesture = None
        last_send_time = 0.0
        heartbeat_interval = cfg.HEARTBEAT_INTERVAL

        while True:
            # Read input from stdin
            if conn is None:
                input_data = input()
            else:
                # Drain the pipe buffer and collect any pending predictions
                input_data = None
                any_received = False
                # Drain all available messages and keep the last prediction; update recent
                while conn.poll():  # Check if data is available without blocking
                    try:
                        d = conn.recv()
                        any_received = True
                        input_data = d  # keep last for backwards-compat
                        timestamp = input_data.get("timestamp")
                        # extract numeric prediction if present and append to recent
                        try:
                            p = int(d.get("prediction", 0))
                        except Exception:
                            p = None
                        if p is not None:
                            recent.append(p)
                    except EOFError:
                        logger.info("Connection closed")
                        return

                now = time.perf_counter()

                # If we received new data, compute smoothed prediction and create input_data
                if any_received and len(recent) > 0:
                    if cfg.SMOOTH_WINDOW > 1:
                        if cfg.SMOOTH_METHOD == 'mode':
                            counts = Counter(recent)
                            most_common = counts.most_common()
                            top_count = most_common[0][1]
                            candidates = [val for val, cnt in most_common if cnt == top_count]
                            # if tie, pick the most recent candidate
                            for v in reversed(recent):
                                if v in candidates:
                                    smoothed_pred = v
                                    break
                        elif cfg.SMOOTH_METHOD == 'mean':
                            smoothed_pred = int(round(mean(recent)))
                        else:
                            smoothed_pred = recent[-1]
                    else:
                        smoothed_pred = recent[-1]

                    input_data = {
                        "prediction": smoothed_pred,
                        "timestamp": timestamp
                    }

                # If no new data was received, optionally send a heartbeat (repeat last gesture)
                if not any_received:
                    # send heartbeat only if user enabled it and we have a last_gesture
                    if heartbeat_interval and last_gesture is not None:
                        if (now - last_send_time) >= heartbeat_interval:
                            try:
                                comm_controller.send_gesture(last_gesture)
                                last_send_time = now
                                logger.debug(f"Heartbeat: re-sent gesture [{last_gesture}]")
                            except Exception as e:
                                logger.error(f"Error sending heartbeat gesture: {e}")
                    # small sleep to avoid busy waiting but keep responsiveness
                    time.sleep(cfg.POLL_SLEEP_DELAY)
                    # continue to next iteration (no new prediction to process)
                    continue

            # Exit the loop if no more input is received
            if input_data is None or input_data == "":
                logger.warning("NO INPUT RECEIVED")
                continue
           
            # print("Communicator Received input:", input_data)
            if input_data == "exit":
                break

            # Process the input 
            try:
                input_pred = int(input_data["prediction"])
                # timestamp = input_data["timestamp"]
                if int(input_pred) not in range(cfg.NUM_CLASSES): 
                    input_pred = 0
                gesture = gjutils.get_label_from_index(input_pred, images, gestures_dict)

                logger.info(f"Input: pred({input_pred})  gest[{gesture}]: {input_data}... received data / sending gesture ...")
            
            except Exception as e:
                logger.error(f"Invalid input. Error: {e}")
                continue

            # Send the gesture to the hand
            try:
                comm_controller.send_gesture(gesture)
                last_gesture = gesture
                last_send_time = time.perf_counter()
            except Exception as e:
                logger.error(f"Error sending gesture: {e}")
            logger.debug("="*50)
            
    except Exception as e:
        logger.error(f"Error communicator: {e}")
    finally:
        comm_controller.disconnect()
        logger.info("Communicator Exiting...")


# CONNECTION HANDLER

def run_process(target, conn: Connection):
    try:
        target(conn)
    except Exception as e:
        logger.error(f"An error occurred in a subprocess: {e}")
    finally:
        conn.close()

def main():
    """
    Main entry point for realtime control.
    """
    from multiprocessing import Process, Pipe
    
    # Create pipes for communication
    conn_p, conn_c = Pipe()
    conn_rp, conn_rc = Pipe()
    
    # Start processes
    prediction_process = Process(target=predicator, args=(True, conn_p))
    communicator_process = Process(target=communicator, args=(conn_c,))
    
    try:
        prediction_process.start()
        communicator_process.start()
        
        prediction_process.join()
        communicator_process.join()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        prediction_process.terminate()
        communicator_process.terminate()
        prediction_process.join()
        communicator_process.join()

if __name__ == "__main__":
    try:
            parent_conn, child_conn = Pipe()
            
            p1 = Process(target=run_process, args=(run_predicator_process, parent_conn))
            p2 = Process(target=run_process, args=(run_controller_process, child_conn))
            
            p1.start()
            p2.start()
            
            p1.join()
            p2.join()
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")
    finally:
        parent_conn.close()
        child_conn.close()

        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()
