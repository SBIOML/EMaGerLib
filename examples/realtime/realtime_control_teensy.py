"""
Real-time EMG control using Psyonic Hand via Teensy Controller.
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
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="realtime_control_teensy")
    save_config_if_requested(args, cfg, script_name="realtime_control_teensy")
    return args, cfg


def run_predictor_thread(ctrl, cfg):
    try:
        logger.info("Starting predictor worker...")

        predicator(
            use_gui=cfg.USE_GUI,
            ctrl_shared=ctrl,
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


def run_controller_thread(ctrl, cfg, args):
    comm_controller = None
    try:
        logger.info("Starting controller worker...")

        time.sleep(3)

        logger.info(f"Connecting to Teensy on port: {args.port or 'auto-detect'}")
        comm_controller = InterfaceControl(
            hand_type="psyonic_teensy",
            cfg=cfg,
            port=args.port,
            baudrate=args.baudrate
        )
        comm_controller.connect()
        logger.info("Teensy controller connected and ready")

        gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
        images = gjutils.get_images_list(cfg.MEDIA_PATH)

        logger.info("Connected to LibEMG shared memory")

        last_sent_gesture = None

        logger.info("Controller ready, waiting for predictions...")
        logger.info(f"Controller poll rate: {cfg.CONTROLLER_POLL_RATE}s")

        while not stop_event.is_set():
            predictions = ctrl.get_data(['predictions'])

            if predictions is None:
                time.sleep(cfg.CONTROLLER_POLL_RATE)
                continue

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

            if gesture != last_sent_gesture:
                try:
                    comm_controller.send_gesture(gesture)
                    last_sent_gesture = gesture
                    logger.info(f"Sent gesture [{gesture}] to Teensy")
                except Exception as e:
                    logger.error(f"Error sending gesture: {e}")

            time.sleep(cfg.CONTROLLER_POLL_RATE)

        logger.info("Controller worker stopped")

    except Exception as e:
        logger.error(f"Error in controller worker: {e}", exc_info=True)
    finally:
        if comm_controller is not None:
            try:
                logger.info("Disconnecting from Teensy...")
                comm_controller.disconnect()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        stop_event.set()


def main(argv=None):
    from libemg.environments.controllers import ClassifierController
    import serial.tools.list_ports

    args, cfg = setup_runtime(argv)

    controller_thread = None
    predictor_thread = None

    try:
        logger.info("=" * 60)
        logger.info("Real-time EMG Control via Psyonic Teensy Controller")
        logger.info("=" * 60)
        logger.info(f"Configuration: {args.config}")
        logger.info(f"GUI enabled: {cfg.USE_GUI}")
        logger.info(f"Predictor speed: {cfg.PREDICTOR_DELAY}s between predictions")
        logger.info(f"Controller speed: {cfg.CONTROLLER_POLL_RATE}s between polls")
        logger.info(f"Teensy port: {args.port or 'auto-detect'}")
        logger.info("=" * 60)

        ports = serial.tools.list_ports.comports()
        logger.info("Available serial ports:")
        for i, port in enumerate(ports):
            logger.info(f"  [{i}] {port.device}: {port.description}")

        logger.info("Connecting to LibEMG shared memory...")
        ctrl = ClassifierController('predictions', cfg.NUM_CLASSES)
        logger.info("✓ LibEMG shared memory connected")

        controller_thread = threading.Thread(
            target=run_controller_thread,
            args=(ctrl, cfg, args),
            daemon=True,
            name="Controller"
        )
        controller_thread.start()
        logger.info("✓ Controller thread started")

        if cfg.USE_GUI:
            logger.info("GUI enabled: running predictor in main thread.")
            try:
                run_predictor_thread(ctrl, cfg)
            except Exception as e:
                logger.error(f"Fatal error in predictor: {e}", exc_info=True)
        else:
            predictor_thread = threading.Thread(
                target=run_predictor_thread,
                args=(ctrl, cfg),
                daemon=True,
                name="Predictor"
            )
            predictor_thread.start()
            logger.info("✓ Predictor thread started")

            logger.info("")
            logger.info("System running. Press Ctrl+C to stop.")
            logger.info("")

            while not stop_event.is_set():
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Keyboard interrupt received - shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Stopping all workers...")
        stop_event.set()

        if predictor_thread is not None and predictor_thread.is_alive():
            predictor_thread.join(timeout=3)
            if predictor_thread.is_alive():
                logger.warning("⚠ Predictor thread did not stop cleanly")
            else:
                logger.info("✓ Predictor thread stopped")

        if controller_thread is not None and controller_thread.is_alive():
            controller_thread.join(timeout=3)
            if controller_thread.is_alive():
                logger.warning("⚠ Controller thread did not stop cleanly")
            else:
                logger.info("✓ Controller thread stopped")

        logger.info("=" * 60)
        logger.info("Application exited")
        logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
