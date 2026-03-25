"""
Real-time EMG control using THREADING with direct shared memory access.
"""

import threading
import time
from pathlib import Path
import logging

from examples.realtime.realtime_prediction import predicator
import emagerlib.utils.gestures_json as gjutils
from emagerlib.control.interface_control import InterfaceControl
from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

logger = logging.getLogger(__name__)
stop_event = threading.Event()


def parse_args(argv=None):
    parser = create_parser(
        description="Real-time EMG prosthetic hand control (Threading with shared memory)",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="realtime_control_threads")
    save_config_if_requested(args, cfg, script_name="realtime_control_threads")
    return args, cfg


def run_predictor_thread(cfg):
    try:
        logger.info("Starting predictor worker...")
        predicator(
            use_gui=cfg.USE_GUI,
            conn=None,
            delay=cfg.PREDICTOR_DELAY,
            timeout_delay=cfg.PREDICTOR_TIMEOUT_DELAY,
            cfg=cfg,
        )

    except Exception as e:
        logger.error(f"Error in predictor worker: {e}", exc_info=True)
    finally:
        stop_event.set()
        logger.info("Predictor worker stopped")


def run_controller_thread(cfg):
    from libemg.environments.controllers import ClassifierController

    comm_controller = None
    try:
        logger.info("Starting controller worker...")

        time.sleep(3)

        comm_controller = InterfaceControl(hand_type="psyonic", cfg=cfg)
        comm_controller.connect()
        logger.info("Hand controller connected")

        gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
        images = gjutils.get_images_list(cfg.MEDIA_PATH)

        ctrl = ClassifierController('predictions', cfg.NUM_CLASSES)
        logger.info("Connected to LibEMG shared memory")

        last_sent_gesture = None

        loop_times = []
        read_times = []
        iteration = 0

        logger.info("Controller ready, waiting for predictions...")

        while not stop_event.is_set():
            loop_start = time.perf_counter()

            read_start = time.perf_counter()
            predictions = ctrl.get_data(['predictions'])
            read_elapsed = time.perf_counter() - read_start
            read_times.append(read_elapsed)

            if predictions is None:
                time.sleep(cfg.CONTROLLER_POLL_RATE)
                continue

            try:
                input_pred = int(predictions[0])
                logger.debug(f"Raw prediction: {input_pred}")

                if input_pred not in range(cfg.NUM_CLASSES):
                    logger.warning(f"Invalid prediction {input_pred}, using REST (0)")
                    input_pred = 0

                gesture = gjutils.get_label_from_index(input_pred, images, gestures_dict)
                logger.info(f"Prediction {input_pred} -> Gesture [{gesture}]")

            except Exception as e:
                logger.error(f"Error processing prediction: {e}", exc_info=True)
                time.sleep(cfg.CONTROLLER_POLL_RATE)
                continue

            if gesture != last_sent_gesture:
                try:
                    comm_controller.send_gesture(gesture)
                    last_sent_gesture = gesture
                    logger.info(f"✓ SENT gesture [{gesture}] to hand")
                except Exception as e:
                    logger.error(f"✗ Error sending gesture: {e}", exc_info=True)
            else:
                logger.debug("Same gesture, skipping send")

            time.sleep(cfg.CONTROLLER_POLL_RATE)

            loop_elapsed = time.perf_counter() - loop_start
            loop_times.append(loop_elapsed)

            iteration += 1
            if iteration % 100 == 0:
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
        if comm_controller is not None:
            comm_controller.disconnect()
        stop_event.set()


def main(argv=None):
    _, cfg = setup_runtime(argv)

    predictor_thread = None
    controller_thread = None

    try:
        logger.info("Starting threaded real-time control...")
        logger.info(f"GUI enabled: {cfg.USE_GUI}")
        logger.info(f"Controller poll rate: {cfg.CONTROLLER_POLL_RATE}s ({1/cfg.CONTROLLER_POLL_RATE:.0f}Hz)")
        logger.info(f"Smooth window: {cfg.SMOOTH_WINDOW}, method: {cfg.SMOOTH_METHOD}")

        predictor_thread = threading.Thread(
            target=run_predictor_thread,
            args=(cfg,),
            daemon=True,
            name="Predictor"
        )
        predictor_thread.start()
        logger.info("Predictor thread started")

        controller_thread = threading.Thread(
            target=run_controller_thread,
            args=(cfg,),
            daemon=True,
            name="Controller"
        )
        controller_thread.start()
        logger.info("Controller thread started")

        while not stop_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Stopping all workers...")
        stop_event.set()

        if predictor_thread is not None and predictor_thread.is_alive():
            predictor_thread.join(timeout=3)
            if predictor_thread.is_alive():
                logger.warning("Predictor thread did not stop cleanly")

        if controller_thread is not None and controller_thread.is_alive():
            controller_thread.join(timeout=3)
            if controller_thread.is_alive():
                logger.warning("Controller thread did not stop cleanly")

        logger.info("Application exiting")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
