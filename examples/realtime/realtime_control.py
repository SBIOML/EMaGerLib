from multiprocessing.connection import Connection
from multiprocessing import Process, Pipe
import time
from collections import deque, Counter
from statistics import mean
from pathlib import Path
import logging

from examples.realtime.realtime_prediction import predicator
import emagerlib.utils.gestures_json as gjutils
from emagerlib.control.interface_control import InterfaceControl
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Real-time EMG prosthetic hand control",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="realtime_control")
    save_config_if_requested(args, cfg, script_name="realtime_control")
    return args, cfg


def run_predicator_process(conn: Connection = None, cfg=None):
    predicator(
        use_gui=cfg.USE_GUI,
        conn=conn,
        delay=cfg.PREDICTOR_DELAY,
        timeout_delay=cfg.PREDICTOR_TIMEOUT_DELAY,
        cfg=cfg,
    )


def run_controller_process(conn: Connection = None, cfg=None):
    comm_controller = None
    try:
        comm_controller = InterfaceControl(hand_type="psyonic", cfg=cfg)
        comm_controller.connect()

        gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
        images = gjutils.get_images_list(cfg.MEDIA_PATH)

        logger.info("Communicator waiting for data...")
        recent = deque(maxlen=cfg.SMOOTH_WINDOW)
        last_sent_gesture = None

        while True:
            if conn is None:
                input_data = input()
            else:
                input_data = None
                any_received = False
                while conn.poll():
                    try:
                        d = conn.recv()
                        any_received = True
                        input_data = d
                        timestamp = input_data.get("timestamp")
                        try:
                            p = int(d.get("prediction", 0))
                        except Exception:
                            p = None
                        if p is not None:
                            recent.append(p)
                    except EOFError:
                        logger.info("Connection closed")
                        return

                if any_received and len(recent) > 0:
                    if cfg.SMOOTH_WINDOW > 1:
                        if cfg.SMOOTH_METHOD == 'mode':
                            counts = Counter(recent)
                            most_common = counts.most_common()
                            top_count = most_common[0][1]
                            candidates = [val for val, cnt in most_common if cnt == top_count]
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

                if not any_received:
                    time.sleep(cfg.CONTROLLER_POLL_RATE)
                    continue

            if input_data is None or input_data == "":
                logger.warning("NO INPUT RECEIVED")
                continue

            if input_data == "exit":
                break

            try:
                input_pred = int(input_data["prediction"])
                if int(input_pred) not in range(cfg.NUM_CLASSES):
                    input_pred = 0
                gesture = gjutils.get_label_from_index(input_pred, images, gestures_dict)

                logger.info(f"Input: pred({input_pred})  gest[{gesture}]: {input_data}... received data / sending gesture ...")

            except Exception as e:
                logger.error(f"Invalid input. Error: {e}")
                time.sleep(cfg.CONTROLLER_POLL_RATE)
                continue

            if gesture != last_sent_gesture:
                try:
                    comm_controller.send_gesture(gesture)
                    last_sent_gesture = gesture
                    logger.info(f"✓ SENT gesture [{gesture}]")
                except Exception as e:
                    logger.error(f"✗ Error sending gesture: {e}", exc_info=True)
            else:
                logger.debug("Same gesture, skipping send")

            time.sleep(cfg.CONTROLLER_POLL_RATE)
            logger.debug("=" * 50)

    except Exception as e:
        logger.error(f"Error communicator: {e}")
    finally:
        if comm_controller is not None:
            comm_controller.disconnect()
        logger.info("Communicator Exiting...")


def run_process(target, conn: Connection, cfg):
    try:
        target(conn, cfg)
    except Exception as e:
        logger.error(f"An error occurred in a subprocess: {e}")
    finally:
        conn.close()


def main(argv=None):
    _, cfg = setup_runtime(argv)

    parent_conn = None
    child_conn = None
    p1 = None
    p2 = None

    try:
        parent_conn, child_conn = Pipe()

        p1 = Process(target=run_process, args=(run_predicator_process, parent_conn, cfg))
        p2 = Process(target=run_process, args=(run_controller_process, child_conn, cfg))

        p1.start()
        p2.start()

        p1.join()
        p2.join()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")
    finally:
        if parent_conn is not None:
            parent_conn.close()
        if child_conn is not None:
            child_conn.close()

        if p1 is not None and p1.is_alive():
            p1.terminate()
        if p2 is not None and p2.is_alive():
            p2.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
