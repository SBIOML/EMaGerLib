import threading
import time
from pathlib import Path
import logging
import sys

from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.utils.streamer_utils import get_emager_streamer

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Real-time EMG gesture prediction",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="realtime_prediction")
    save_config_if_requested(args, cfg, script_name="realtime_prediction")
    return args, cfg


def update_labels_process(
    stop_event,
    gui,
    ctrl,
    cfg,
    gjutils,
    conn=None,
    delay=None,
    timeout_delay=None,
):
    if delay is None:
        delay = cfg.PREDICTOR_DELAY
    if timeout_delay is None:
        timeout_delay = cfg.PREDICTOR_TIMEOUT_DELAY

    gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
    images = gjutils.get_images_list(cfg.MEDIA_PATH)

    last_prediction = None
    last_sent_time = 0

    while not stop_event.is_set():
        try:
            predictions = ctrl.get_data(['predictions'])
            if predictions is None:
                time.sleep(delay)
                continue

            index = int(predictions[0])

            if index < 0 or index >= cfg.NUM_CLASSES:
                logger.warning(f"Invalid prediction index {index}, skipping")
                time.sleep(delay)
                continue

            current_time = time.time()
            should_forward = not (
                index == last_prediction and (current_time - last_sent_time) < timeout_delay
            )

            ts = current_time
            timestamp = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts - int(ts)) * 1000):03d}"
            output_data = {
                "prediction": index,
                "timestamp": timestamp
            }

            label = gjutils.get_label_from_index(index, images, gestures_dict)

            if label is None:
                logger.warning(f"Failed to get label for index {index}")
                time.sleep(delay)
                continue

            gui.update_label(label)

            if should_forward:
                logger.info(f"Prediction: {index} ({label})")

            if conn is not None and should_forward:
                logger.info(f"Output : pred({predictions[0]})  gest[{label}] {output_data}... sending data ...")
                conn.send(output_data)
                last_prediction = index
                last_sent_time = current_time

            if conn is None and should_forward:
                last_prediction = index
                last_sent_time = current_time

            time.sleep(delay)

        except Exception as e:
            logger.error(f"Error in update_labels_process loop: {e}", exc_info=True)
            time.sleep(delay)


def predicator(
    use_gui=True,
    ctrl_shared=None,
    conn=None,
    delay=None,
    timeout_delay=None,
    cfg=None,
):
    import numpy as np
    import torch
    from multiprocessing import Lock
    from libemg.data_handler import OnlineDataHandler
    from libemg.emg_predictor import EMGClassifier, OnlineEMGClassifier
    from libemg.feature_extractor import FeatureExtractor
    from libemg.filtering import Filter
    from libemg.environments.controllers import ClassifierController

    import emagerlib.models.models as etm
    from emagerlib.visualization.realtime_gui import RealTimeGestureUi
    import emagerlib.utils.gestures_json as gjutils

    if cfg is None:
        _, cfg = setup_runtime()

    if delay is None:
        delay = cfg.PREDICTOR_DELAY
    if timeout_delay is None:
        timeout_delay = cfg.PREDICTOR_TIMEOUT_DELAY

    p, smi = get_emager_streamer(cfg.EMAGER_VERSION)
    smi_emg = [item for item in smi if item[0] in ['emg', 'emg_count']]
    logger.info(f"Streamer created: process: {p}, smi : {smi_emg}")
    odh = OnlineDataHandler(shared_memory_items=smi_emg)

    filter = Filter(cfg.SAMPLING)
    notch_filter_dictionary = {"name": "notch", "cutoff": 60, "bandwidth": 3}
    filter.install_filters(notch_filter_dictionary)
    bandpass_filter_dictionary = {"name": "bandpass", "cutoff": [20, 450], "order": 4}
    filter.install_filters(bandpass_filter_dictionary)
    odh.install_filter(filter)
    logger.info("Data handler created")

    fe = FeatureExtractor()
    fg = ["MAV"]
    logger.info(f"Feature group: {fg}")

    model = etm.EmagerCNN((4, 16), cfg.NUM_CLASSES, -1)

    if cfg.MODEL_PATH is None:
        logger.error(f"No model found in {cfg.SESSION_PATH}. Please train a model first or specify MODEL_NAME in config.")
        logger.error("You can train a model using: emager-train-cnn -c <config_file>")
        sys.exit(1)

    logger.info(f"Loading model from: {cfg.MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(cfg.MODEL_PATH))
        model.eval()
    except RuntimeError as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Model file not found: {cfg.MODEL_PATH}")
        sys.exit(1)

    classi = EMGClassifier(model)
    classi.add_majority_vote(cfg.MAJORITY_VOTE)

    smm_items = [
        ["classifier_output", (100, 4), np.double, Lock()],
        ['classifier_input', (100, 1 + 64), np.double, Lock()],
    ]
    oclassi = OnlineEMGClassifier(
        classi,
        cfg.WINDOW_SIZE,
        cfg.WINDOW_INCREMENT,
        odh,
        fg,
        std_out=False,
        smm=True,
        smm_items=smm_items,
    )

    files = gjutils.get_images_list(cfg.MEDIA_PATH)
    logger.debug(f"Files: {files}")
    logger.info("Creating GUI...")
    gui = RealTimeGestureUi(files)

    if ctrl_shared is not None:
        ctrl = ctrl_shared
        logger.debug("Using shared ClassifierController")
    else:
        ctrl = ClassifierController('predictions', cfg.NUM_CLASSES)
        logger.debug("Created new ClassifierController (standalone mode)")

    stop_event = threading.Event()
    update_label_process = threading.Thread(
        target=update_labels_process,
        args=(stop_event, gui, ctrl, cfg, gjutils, conn, delay, timeout_delay),
    )

    try:
        logger.info("Starting classification...")
        oclassi.run(block=False)
        logger.info("Starting process thread...")
        update_label_process.start()
        logger.info("Starting GUI...")
        if use_gui:
            gui.run()
        else:
            while True:
                time.sleep(1)

    except Exception as e:
        logger.error(f"Error during classification: {e}")

    finally:
        if conn is not None:
            conn.send("exit")
        stop_event.set()
        oclassi.stop_running()
        logger.info("Exiting")


def main(argv=None):
    _, cfg = setup_runtime(argv)
    predicator(
        use_gui=True,
        ctrl_shared=None,
        conn=None,
        delay=cfg.PREDICTOR_DELAY,
        timeout_delay=cfg.PREDICTOR_TIMEOUT_DELAY,
        cfg=cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
