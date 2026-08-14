"""Run the gesture model on a microcontroller and display its predictions.

The PC is a wire here, not a processor. It forwards the base station's raw USB
stream to the MCU byte for byte, and displays the predictions the MCU sends
back. Frame decoding, filtering, feature extraction and inference all happen in
the firmware (separate repository); the wire format for the return path is
specified in `docs/MCU_PROTOCOL.md`.

    base station --USB--> PC (relay, verbatim) --USB--> MCU
                          PC <--predictions-- MCU

Usage
-----
    emager mcu-predict --mcu-port COM7
    emager mcu-predict --mcu-port /dev/ttyACM0 --no-gui
    emager mcu-predict --mcu-port COM7 --record capture.bin   # keep the raw stream
    emager mcu-predict --mcu-port COM7 --replay capture.bin   # no armband needed
    emager mcu-predict --list-ports

Note this script never imports libemg or torch, so the Windows Qt/torch DLL
ordering trap described in CLAUDE.md does not apply.
"""

import logging
import threading
import time
from pathlib import Path

import serial
import serial.tools.list_ports

from emagerlib.config.load_config import load_config
from emagerlib.mcu.prediction_reader import PredictionReader
from emagerlib.mcu.relay import (
    BASESTATION_BAUD,
    BASESTATION_VID_PID,
    FileSource,
    Relay,
    SerialSource,
)
from emagerlib.utils.arg_parser import create_parser, save_config_if_requested, setup_logging

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = create_parser(
        description="Relay EMaGer data to an MCU and display its predictions",
        default_config=str(DEFAULT_CONFIG),
    )
    parser.add_argument("--mcu-port", default=None,
                        help="Serial port of the MCU (overrides MCU_PORT in the config)")
    parser.add_argument("--mcu-baud", type=int, default=None,
                        help="Baud rate for the MCU link (overrides MCU_BAUD; "
                             "nominal over USB CDC, must sustain ~195 kB/s over a real UART)")
    parser.add_argument("--basestation-port", default=None,
                        help="Serial port of the base station (default: auto-detect by VID/PID)")
    parser.add_argument("--record", default=None, metavar="FILE",
                        help="Also write the raw relayed stream to FILE, for later --replay")
    parser.add_argument("--replay", default=None, metavar="FILE",
                        help="Replay a recorded stream instead of reading the base station")
    parser.add_argument("--fast", action="store_true",
                        help="With --replay, send as fast as the link allows instead of "
                             "pacing to the real frame rate")
    parser.add_argument("--loop", action="store_true",
                        help="With --replay, restart the capture when it ends")
    parser.add_argument("--no-gui", action="store_true",
                        help="Log predictions to the console instead of opening the gesture UI")
    parser.add_argument("--no-stats", action="store_true",
                        help="Disable the passive frame-health counters on the relayed stream")
    parser.add_argument("--list-ports", action="store_true",
                        help="List available serial ports and exit")
    return parser.parse_args(argv)


def setup_runtime(argv=None, args=None):
    if args is None:
        args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="mcu_predict")
    save_config_if_requested(args, cfg, script_name="mcu_predict")
    return args, cfg


def list_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for p in ports:
        tag = "  <- base station" if (p.vid, p.pid) == BASESTATION_VID_PID else ""
        print(f"  {p.device}: {p.description or '<no description>'} "
              f"(VID: {p.vid}, PID: {p.pid}){tag}")


def build_source(args, cfg):
    """Live base station, or a recorded capture."""
    if args.replay:
        path = Path(args.replay)
        if not path.exists():
            raise SystemExit(f"Capture not found: {path}")
        return FileSource(path, realtime=not args.fast, loop=args.loop)

    return SerialSource(
        port=args.basestation_port or cfg.BASESTATION_PORT,
        baud=cfg.BASESTATION_BAUD or BASESTATION_BAUD,
        vid_pid=BASESTATION_VID_PID,
    )


def make_display(cfg, use_gui):
    """Return ``(show_fn, run_fn, stop_fn)`` for whichever display was asked for.

    ``run_fn`` blocks until the user quits; with --no-gui it just idles, so the
    caller's shutdown path is the same either way.
    """
    import emagerlib.utils.gestures_json as gjutils

    gestures_dict = gjutils.get_gestures_dict(cfg.MEDIA_PATH)
    images = gjutils.get_images_list(cfg.MEDIA_PATH)

    gui = None
    if use_gui:
        # Imported here so --no-gui works on machines with no Qt/display.
        from emagerlib.visualization.realtime_gui import RealTimeGestureUi
        gui = RealTimeGestureUi(images)

    def show(pred):
        if not pred.valid:
            logger.warning("MCU reports no prediction (class -1) -- is it calibrated?")
            return
        label = gjutils.get_label_from_index(pred.class_id, images, gestures_dict)
        if label is None:
            logger.warning(f"No label for class index {pred.class_id}")
            return
        if gui is not None:
            gui.update_label(label)
        return label

    stop_event = threading.Event()

    def run():
        if gui is not None:
            gui.run()
        else:
            while not stop_event.is_set():
                time.sleep(0.1)

    return show, run, stop_event.set


def main(argv=None):
    args = parse_args(argv)
    if args.list_ports:
        # Answered before touching the config: it is what you run when nothing
        # is set up yet.
        list_ports()
        return 0

    args, cfg = setup_runtime(args=args)

    mcu_port = args.mcu_port or cfg.MCU_PORT
    if not mcu_port:
        logger.error("No MCU port. Pass --mcu-port or set MCU_PORT in the config.")
        list_ports()
        return 1
    mcu_baud = args.mcu_baud or cfg.MCU_BAUD or 2_000_000

    try:
        source = build_source(args, cfg)
    except RuntimeError as exc:
        logger.error(exc)
        return 1

    relay = Relay(
        source=source,
        dst_port=mcu_port,
        dst_baud=mcu_baud,
        record_path=Path(args.record) if args.record else None,
        stats=not args.no_stats,
    )

    show, run_display, stop_display = make_display(cfg, use_gui=not args.no_gui)

    last_class = None

    def on_prediction(pred):
        nonlocal last_class
        label = show(pred)
        if pred.class_id != last_class:
            logger.info(f"Prediction: class {pred.class_id}"
                        + (f" ({label})" if label is not None else "")
                        + (f"  [{pred.infer_us} us on device]" if pred.infer_us else "")
                        + f"  frame {pred.frame_id}")
            last_class = pred.class_id

    try:
        relay.open()
    except (serial.SerialException, OSError) as exc:
        logger.error(f"Could not open the link: {exc}")
        list_ports()
        return 1

    # Share the relay's port: on a single bidirectional CDC endpoint the relay
    # writes the EMG stream while the reader pulls predictions off the same
    # handle. Point this at a second port if the MCU exposes one.
    reader = PredictionReader(relay.dst, on_prediction=on_prediction)

    stop_event = threading.Event()
    relay_thread = threading.Thread(
        target=_run_relay, args=(relay, stop_event, stop_display),
        name="relay", daemon=True,
    )

    try:
        reader.start()
        relay_thread.start()
        logger.info("Relaying. Ctrl-C to stop.")
        run_display()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        stop_event.set()
        relay_thread.join(timeout=2.0)
        reader.stop()
        relay.close()
        logger.info(f"MCU link: {reader.summary()}")

    return 0


def _run_relay(relay, stop_event, on_finished):
    try:
        relay.run(stop_event)
    except Exception as exc:
        logger.error(f"Relay stopped: {exc}", exc_info=True)
    finally:
        # A finished replay should close the UI rather than leave it frozen on
        # the last gesture with nothing arriving.
        on_finished()


if __name__ == "__main__":
    raise SystemExit(main())
