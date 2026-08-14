"""Byte-for-byte relay from the EMaGer base station to the MCU.

The MCU must see exactly the same byte stream whether the base station is wired
to it directly or routed through the PC. So this module does not decode,
reframe, filter, or re-timestamp anything -- it reads bytes and writes the same
bytes. The frame accounting below runs on a *copy*, off the data path, purely so
a human can tell whether the link is healthy.

One consequence worth stating: the base station's serial port is exclusive.
libemg's ``EmagerV3Streamer`` opens it in its own process and does not expose
raw bytes, so the relay and libemg cannot both have it. In this setup the PC
runs no inference at all -- the MCU does -- so the relay simply takes the port
and libemg is not involved.
"""

import logging
import platform
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

# Matches libemg's Emager3 defaults (libemg/_streamers/_emagerv3_streamer.py).
# Duplicated rather than imported: importing libemg pulls in torch, which is a
# lot of machinery for a program whose job is to copy bytes.
BASESTATION_VID_PID = (12259, 256)
BASESTATION_BAUD = 3_000_000

# EMaGer v3 frame geometry, used ONLY for the passive health counters. The
# authoritative definition is Emager3's docstring in libemg. The v3 cuff and the
# 32-channel Puck share all of these -- they differ only in payload packing,
# which the relay never looks at.
FRAME_SIZE = 8192
FRAME_HEADER = b"\xAA\x55"
FRAME_TRAILER = b"\x55\xAA"
TRAILER_OFFSET = 8190
COUNTER_OFFSET = 8176
SAMPLES_PER_FRAME = 84

# 2000 Hz / 84 samples per frame ~= 23.8 frames/s ~= 195 kB/s. Used to pace
# replay, and to size the "is the MCU link fast enough" note in the docs.
FRAME_PERIOD_S = SAMPLES_PER_FRAME / 2000.0


def find_port(vid_pid: Tuple[int, int], description: Optional[str] = None) -> str:
    """Locate a serial port by USB VID/PID, or by a substring of its description.

    Mirrors the lookup in libemg's ``Emager3.__init__`` so the relay finds the
    same device libemg would, including the macOS ``cu`` -> ``tty`` fixup.
    """
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if description is not None:
            match = description in (p.description or "")
        else:
            match = (p.vid, p.pid) == tuple(vid_pid)
        if match:
            return p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")

    available = "\n".join(
        f"  - {getattr(p, 'device', None) or p.name}: {p.description or '<no description>'} "
        f"(VID: {p.vid}, PID: {p.pid})"
        for p in ports
    ) or "  (no serial ports found)"
    wanted = f"description containing {description!r}" if description else f"VID/PID {vid_pid}"
    raise RuntimeError(f"No serial port matching {wanted}. Available ports:\n{available}")


class FrameStats:
    """Passive health counters for the relayed stream.

    Deliberately never touches the relayed bytes: it is fed a copy and its only
    output is numbers for a human. A parsing bug here must not be able to alter
    what the MCU receives.
    """

    def __init__(self):
        self._buf = bytearray()
        self.bytes_seen = 0
        self.frames_ok = 0
        self.bad_trailer = 0
        self.counter_gaps = 0
        self.lost_frames = 0
        self._last_counter: Optional[int] = None

    def feed(self, data: bytes) -> None:
        self.bytes_seen += len(data)
        self._buf += data

        pos = 0
        while True:
            h = self._buf.find(FRAME_HEADER, pos)
            if h < 0:
                # Keep a frame's worth minus one byte: a header could straddle
                # this chunk boundary.
                keep = min(len(self._buf), FRAME_SIZE - 1)
                del self._buf[:len(self._buf) - keep]
                return
            if len(self._buf) - h < FRAME_SIZE:
                del self._buf[:h]
                return

            if self._buf[h + TRAILER_OFFSET:h + TRAILER_OFFSET + 2] == FRAME_TRAILER:
                self.frames_ok += 1
                counter = int.from_bytes(
                    self._buf[h + COUNTER_OFFSET:h + COUNTER_OFFSET + 4], "big"
                )
                if self._last_counter is not None:
                    expected = (self._last_counter + 1) & 0xFFFFFFFF
                    if counter != expected:
                        self.counter_gaps += 1
                        self.lost_frames += (counter - expected) & 0xFFFFFFFF
                self._last_counter = counter
                pos = h + FRAME_SIZE
            else:
                self.bad_trailer += 1
                pos = h + 1

            if pos > FRAME_SIZE * 2:
                del self._buf[:pos]
                pos = 0

    def summary(self) -> str:
        return (f"{self.frames_ok} frames, {self.bytes_seen/1024:.0f} KiB, "
                f"{self.bad_trailer} bad trailers, {self.counter_gaps} counter gaps "
                f"({self.lost_frames} frames lost)")


class SerialSource:
    """Live base station: non-blocking reads, we do our own buffering."""

    def __init__(self, port: Optional[str] = None, baud: int = BASESTATION_BAUD,
                 vid_pid: Tuple[int, int] = BASESTATION_VID_PID,
                 description: Optional[str] = None):
        self.port = port or find_port(vid_pid, description)
        self.baud = baud
        self.ser: Optional[serial.Serial] = None

    def open(self) -> None:
        logger.info(f"Base station: opening {self.port} at {self.baud} baud")
        self.ser = serial.Serial(self.port, self.baud, timeout=0)
        # Start from a clean buffer: whatever accumulated before we attached is
        # stale, and forwarding it would hand the MCU a torn frame plus data
        # from an arbitrary point in the past.
        self.ser.reset_input_buffer()

    def read_chunk(self) -> bytes:
        n = self.ser.in_waiting
        return self.ser.read(n) if n else b""

    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()


class FileSource:
    """Replay a previously recorded raw capture.

    Same bytes, same order, so the MCU cannot tell the difference -- which makes
    a firmware bug reproducible instead of needing the armband on someone's arm.
    """

    def __init__(self, path: Path, chunk_size: int = FRAME_SIZE, realtime: bool = True,
                 loop: bool = False):
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.realtime = realtime
        self.loop = loop
        self._fh = None
        self._next_due = 0.0
        self.exhausted = False

    def open(self) -> None:
        size = self.path.stat().st_size
        logger.info(f"Replay: {self.path} ({size/1024:.0f} KiB, ~{size/FRAME_SIZE:.0f} frames, "
                    f"{'real-time' if self.realtime else 'as fast as possible'})")
        self._fh = self.path.open("rb")
        self._next_due = time.monotonic()

    def read_chunk(self) -> bytes:
        if self._fh is None or self.exhausted:
            return b""
        if self.realtime:
            # Pace to the real frame period. Returning empty rather than
            # sleeping keeps the relay loop responsive to its stop event.
            now = time.monotonic()
            if now < self._next_due:
                return b""
            self._next_due += FRAME_PERIOD_S * (self.chunk_size / FRAME_SIZE)

        chunk = self._fh.read(self.chunk_size)
        if not chunk:
            if self.loop:
                self._fh.seek(0)
                chunk = self._fh.read(self.chunk_size)
            else:
                self.exhausted = True
                logger.info("Replay: end of capture")
        return chunk

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class Relay:
    """Pumps bytes from a source to the MCU, unmodified.

    Parameters
    ----------
    source:
        Anything with ``open()`` / ``read_chunk()`` / ``close()`` --
        :class:`SerialSource` live, or :class:`FileSource` for replay.
    dst_port, dst_baud:
        The MCU's serial port. Over USB CDC the baud rate is nominal; over a
        real UART it must sustain ~195 kB/s, i.e. at least 2 Mbaud.
    record_path:
        Optional file to also write the raw stream to, for later replay.
    stats:
        Enable the passive frame counters.
    """

    def __init__(self, source, dst_port: str, dst_baud: int = 2_000_000,
                 record_path: Optional[Path] = None, stats: bool = True):
        self.source = source
        self.dst_port = dst_port
        self.dst_baud = dst_baud
        self.record_path = Path(record_path) if record_path else None
        self.dst: Optional[serial.Serial] = None
        self._recorder = None

        self.bytes_forwarded = 0
        self.write_timeouts = 0

        self.stats = FrameStats() if stats else None
        self._stats_q: Optional[queue.Queue] = queue.Queue(maxsize=256) if stats else None
        self._stats_thread: Optional[threading.Thread] = None
        self._stats_stop = threading.Event()
        self.stats_dropped_chunks = 0

    def open(self) -> None:
        self.source.open()
        logger.info(f"MCU: opening {self.dst_port} at {self.dst_baud} baud")
        # A write timeout matters: if the MCU stops draining its endpoint, an
        # untimed write blocks the relay forever with no diagnostic.
        self.dst = serial.Serial(self.dst_port, self.dst_baud, timeout=0, write_timeout=1.0)

        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self._recorder = self.record_path.open("wb")
            logger.info(f"Recording raw stream to {self.record_path}")

        if self._stats_q is not None:
            self._stats_thread = threading.Thread(
                target=self._stats_loop, name="relay-stats", daemon=True
            )
            self._stats_thread.start()

    def _stats_loop(self) -> None:
        while not self._stats_stop.is_set():
            try:
                chunk = self._stats_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.stats.feed(chunk)
            except Exception as exc:  # never let diagnostics take down the relay
                logger.debug(f"Frame stats error (ignored): {exc}")

    def pump(self) -> int:
        """Move whatever is available. Returns the number of bytes forwarded."""
        chunk = self.source.read_chunk()
        if not chunk:
            return 0

        try:
            self.dst.write(chunk)
        except serial.SerialTimeoutException:
            # Dropping is the honest failure here: buffering would silently
            # build unbounded latency between the arm and the prediction.
            self.write_timeouts += 1
            logger.warning(
                f"MCU write timed out, dropped {len(chunk)} bytes "
                f"(timeout #{self.write_timeouts}) -- the MCU is not draining the link"
            )
            return 0

        self.bytes_forwarded += len(chunk)

        if self._recorder is not None:
            self._recorder.write(chunk)

        if self._stats_q is not None:
            try:
                self._stats_q.put_nowait(chunk)
            except queue.Full:
                # Diagnostics are expendable; the byte stream is not.
                self.stats_dropped_chunks += 1

        return len(chunk)

    def run(self, stop_event: threading.Event) -> None:
        """Pump until stopped, or until a replay source runs out."""
        while not stop_event.is_set():
            if self.pump() == 0:
                if getattr(self.source, "exhausted", False):
                    logger.info("Source exhausted, stopping relay")
                    return
                # Same 1 ms backoff libemg's streamer uses when idle.
                time.sleep(0.001)

    def close(self) -> None:
        self._stats_stop.set()
        if self._stats_thread is not None:
            self._stats_thread.join(timeout=1.0)
        try:
            self.source.close()
        finally:
            if self.dst is not None and self.dst.is_open:
                self.dst.close()
            if self._recorder is not None:
                self._recorder.close()

        logger.info(f"Relay forwarded {self.bytes_forwarded/1024:.0f} KiB")
        if self.stats is not None:
            logger.info(f"Stream health: {self.stats.summary()}")
            if self.stats_dropped_chunks:
                logger.debug(f"{self.stats_dropped_chunks} chunks skipped by the stats thread "
                             f"(diagnostics only, relayed data unaffected)")
