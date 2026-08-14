"""Background reader for the MCU's prediction stream."""

import logging
import threading
import time
from typing import Callable, List, Optional, Union

import serial

from emagerlib.mcu.protocol import (
    Info,
    LogMessage,
    PacketDecoder,
    Prediction,
    Status,
)

logger = logging.getLogger(__name__)


class PredictionReader:
    """Reads MCU packets on a background thread and hands out predictions.

    The port may be shared with :class:`~emagerlib.mcu.relay.Relay`: on a single
    bidirectional USB CDC endpoint the relay writes while this reads, which is
    the same full-duplex arrangement ``PsyonicTeensyController`` already uses in
    this repo. Pass a separate port only if the MCU exposes two.

    Parameters
    ----------
    port:
        A port name to open, or an already-open port object to share. Anything
        that is not a string is taken to be already open, which keeps this
        testable against an in-memory fake.
    baud:
        Only used when opening a port by name.
    on_prediction:
        Called from the reader thread for each prediction. Keep it cheap.
    """

    def __init__(self, port: Union[str, serial.Serial], baud: int = 2_000_000,
                 on_prediction: Optional[Callable[[Prediction], None]] = None,
                 poll_interval: float = 0.001):
        self._own_serial = isinstance(port, str)
        self._port = port
        self._baud = baud
        self.ser = None if self._own_serial else port
        self.on_prediction = on_prediction
        self.poll_interval = poll_interval

        self.decoder = PacketDecoder()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._latest: Optional[Prediction] = None
        self.info: Optional[Info] = None
        self.status: Optional[Status] = None

        self.n_predictions = 0
        self.seq_gaps = 0
        self.lost_predictions = 0
        self._last_seq: Optional[int] = None

    @property
    def latest(self) -> Optional[Prediction]:
        """The most recent prediction, or None if none has arrived."""
        with self._lock:
            return self._latest

    def start(self) -> None:
        if self._own_serial:
            logger.info(f"MCU predictions: opening {self._port} at {self._baud} baud")
            self.ser = serial.Serial(self._port, self._baud, timeout=0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mcu-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._own_serial and self.ser is not None and self.ser.is_open:
            self.ser.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                n = self.ser.in_waiting
                data = self.ser.read(n) if n else b""
            except (serial.SerialException, OSError) as exc:
                logger.error(f"MCU read failed: {exc}")
                return

            if not data:
                time.sleep(self.poll_interval)
                continue

            for msg in self.decoder.feed(data):
                self._dispatch(msg)

    def _dispatch(self, msg) -> None:
        if isinstance(msg, Prediction):
            self._on_prediction(msg)
        elif isinstance(msg, Info):
            self.info = msg
            logger.info(f"MCU info: firmware={msg.firmware or '<unnamed>'} "
                        f"classes={msg.n_classes} embed_dim={msg.embed_dim} "
                        f"model_hash=0x{msg.model_hash:08X}")
        elif isinstance(msg, Status):
            self.status = msg
            logger.info(f"MCU status: {msg.frames_ok} frames ok, {msg.frames_bad} bad, "
                        f"{msg.resyncs} resyncs, {msg.predictions} predictions, "
                        f"uptime {msg.uptime_ms/1000:.1f}s")
        elif isinstance(msg, LogMessage):
            logger.info(f"[MCU] {msg.text}")

    def _on_prediction(self, pred: Prediction) -> None:
        self.n_predictions += 1

        if self._last_seq is not None:
            expected = (self._last_seq + 1) & 0xFFFFFFFF
            if pred.seq != expected:
                self.seq_gaps += 1
                self.lost_predictions += (pred.seq - expected) & 0xFFFFFFFF
                logger.debug(f"Prediction sequence gap: expected {expected}, got {pred.seq}")
        self._last_seq = pred.seq

        with self._lock:
            self._latest = pred

        if self.on_prediction is not None:
            try:
                self.on_prediction(pred)
            except Exception as exc:
                logger.error(f"on_prediction callback failed: {exc}", exc_info=True)

    def summary(self) -> str:
        d = self.decoder
        parts: List[str] = [
            f"{self.n_predictions} predictions",
            f"{d.packets_ok} packets ok",
        ]
        if d.crc_errors:
            parts.append(f"{d.crc_errors} CRC errors")
        if d.resyncs:
            parts.append(f"{d.resyncs} resyncs")
        if self.seq_gaps:
            parts.append(f"{self.seq_gaps} sequence gaps ({self.lost_predictions} lost)")
        if d.dropped_bytes:
            parts.append(f"{d.dropped_bytes} bytes discarded")
        return ", ".join(parts)
