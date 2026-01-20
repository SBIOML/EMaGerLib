import utils.utils as eutils
import utils.gestures_json as gjutils
from libemg_realtime_prediction import predicator
from control.interface_control import InterfaceControl

from multiprocessing.connection import Connection
from multiprocessing import Lock, Process, Pipe
from config_emager import *
import time
from collections import deque, Counter
from statistics import mean

eutils.set_logging()

# PREDICTOR
def run_predicator_process(conn: Connection=None):
    predicator(use_gui=USE_GUI, conn=conn, delay=PREDICTOR_DELAY, timeout_delay=PREDICTOR_TIMEOUT_DELAY)


# COMMUNICATOR
def run_controller_process(conn: Connection=None):
    try:
        
        comm_controller = InterfaceControl(hand_type="psyonic")
        comm_controller.connect()
        
        gestures_dict = gjutils.get_gestures_dict(MEDIA_PATH)
        images = gjutils.get_images_list(MEDIA_PATH)
        
        # Main loop to read input from stdin
        print("Communicator waiting for data...")
        recent = deque(maxlen=SMOOTH_WINDOW)
        last_gesture = None
        last_send_time = 0.0
        heartbeat_interval = HEARTBEAT_INTERVAL if 'HEARTBEAT_INTERVAL' in globals() else None

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
                        print("Connection closed")
                        return

                now = time.perf_counter()

                # If we received new data, compute smoothed prediction and create input_data
                if any_received and len(recent) > 0:
                    if SMOOTH_WINDOW > 1:
                        if SMOOTH_METHOD == 'mode':
                            counts = Counter(recent)
                            most_common = counts.most_common()
                            top_count = most_common[0][1]
                            candidates = [val for val, cnt in most_common if cnt == top_count]
                            # if tie, pick the most recent candidate
                            for v in reversed(recent):
                                if v in candidates:
                                    smoothed_pred = v
                                    break
                        elif SMOOTH_METHOD == 'mean':
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
                                print(f"Heartbeat: re-sent gesture [{last_gesture}]")
                            except Exception as e:
                                print(f"Error sending heartbeat gesture: {e}")
                    # small sleep to avoid busy waiting but keep responsiveness
                    time.sleep(POLL_SLEEP_DELAY)
                    # continue to next iteration (no new prediction to process)
                    continue

            # Exit the loop if no more input is received
            if input_data is None or input_data == "":
                print("NO INPUT RECEIVED")
                continue
           
            # print("Communicator Received input:", input_data)
            if input_data == "exit":
                break

            # Process the input 
            try:
                input_pred = int(input_data["prediction"])
                # timestamp = input_data["timestamp"]
                if int(input_pred) not in range(NUM_CLASSES): 
                    input_pred = 0
                gesture = gjutils.get_label_from_index(input_pred, images, gestures_dict)

                print(f"Input: pred({input_pred})  gest[{gesture}]: {input_data}" + " "*10 + "... received data /  sending gesture ...")
            
            except Exception as e:
                print("Invalid input. Error: ", e)
                continue

            # Send the gesture to the hand
            try:
                comm_controller.send_gesture(gesture)
                last_gesture = gesture
                last_send_time = time.perf_counter()
            except Exception as e:
                print(f"Error sending gesture: {e}")
            print("="*50)
            
    except Exception as e:
        print(f"Error communicator: {e}")
    finally:
        comm_controller.disconnect()
        print("Communicator Exiting...")


# CONNECTION HANDLER

def run_process(target, conn: Connection):
    try:
        target(conn)
    except Exception as e:
        print(f"An error occurred in a subprocess: {e}")
    finally:
        conn.close()

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
        print(f"An error occurred in the main process: {e}")
    finally:
        parent_conn.close()
        child_conn.close()

        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()
