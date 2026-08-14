"""Wire format for the MCU -> PC link.

This is the *return* path only. The PC -> MCU direction carries the base
station's raw byte stream and nothing else -- adding control packets to it would
make the stream differ from what the MCU sees when the base station is wired
directly, which is the one property this whole setup exists to preserve. So the
MCU talks, the PC listens, and there is no host-to-device channel here.

Packet layout (little-endian throughout)::

    offset  size  field
    0       2     SOF = 0xA5 0x5A
    2       1     protocol version (currently 1)
    3       1     message type (see MsgType)
    4       2     payload length, uint16
    6       N     payload
    6+N     2     CRC-16/CCITT-FALSE over bytes [2 .. 6+N-1]

Little-endian is deliberate: both ends are little-endian machines, so neither
has to byte-swap. Note this differs from the EMaGer frame counter, which is
big-endian on the wire -- the firmware byte-swaps it once when it reads the
frame, then everything downstream stays native.

The decoder resynchronizes the way libemg's ``Emager3`` parser does: scan for
the start-of-frame marker, sanity-check the header, verify the CRC, and on any
failure advance a single byte and rescan. A listener that attaches mid-stream,
or after the MCU has printed boot chatter, converges on the next valid packet.

The full specification, including a reference C implementation for the firmware
side, is in ``docs/MCU_PROTOCOL.md``.
"""

import logging
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

SOF0 = 0xA5
SOF1 = 0x5A
SOF = bytes((SOF0, SOF1))

PROTOCOL_VERSION = 1

HEADER_SIZE = 6
CRC_SIZE = 2
OVERHEAD = HEADER_SIZE + CRC_SIZE

# Bounds the decoder's work on a garbage stream, and tells the firmware author
# how large a transmit buffer needs to be. The largest real packet is a
# PREDICTION with a score per class, which is nowhere near this.
MAX_PAYLOAD = 512


class MsgType(IntEnum):
    """Message type byte."""

    INFO = 0x01
    PREDICTION = 0x02
    STATUS = 0x03
    LOG = 0x7F


# ---------------------------------------------------------------------------
#  CRC-16/CCITT-FALSE  (poly 0x1021, init 0xFFFF, no reflection, no final xor)
# ---------------------------------------------------------------------------

def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE. Table-free so the firmware can copy it as-is."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# ---------------------------------------------------------------------------
#  Payloads
# ---------------------------------------------------------------------------

# INFO: n_classes u8, embed_dim u16, model_hash u32, fw_len u8, then fw_len ASCII bytes
_INFO_FMT = "<BHIB"
_INFO_SIZE = struct.calcsize(_INFO_FMT)

# PREDICTION: frame_id u32, seq u32, class_id i8, n_scores u8, infer_us u16,
#             then n_scores float32
_PRED_FMT = "<IIbBH"
_PRED_SIZE = struct.calcsize(_PRED_FMT)

# STATUS: frames_ok u32, frames_bad u32, resyncs u32, predictions u32, uptime_ms u32
_STATUS_FMT = "<IIIII"
_STATUS_SIZE = struct.calcsize(_STATUS_FMT)


@dataclass
class Info:
    """Model/firmware identification, emitted at boot and periodically.

    Periodic re-emission matters because there is no host-to-device channel: a
    PC that attaches after the MCU booted has no way to ask, so the MCU has to
    volunteer it.
    """

    n_classes: int
    embed_dim: int
    model_hash: int
    firmware: str = ""

    def encode(self) -> bytes:
        fw = self.firmware.encode("ascii", errors="replace")
        if len(fw) > 255:
            fw = fw[:255]
        return struct.pack(_INFO_FMT, self.n_classes, self.embed_dim,
                           self.model_hash, len(fw)) + fw

    @classmethod
    def decode(cls, payload: bytes) -> "Info":
        if len(payload) < _INFO_SIZE:
            raise ValueError(f"INFO payload too short: {len(payload)} < {_INFO_SIZE}")
        n_classes, embed_dim, model_hash, fw_len = struct.unpack_from(_INFO_FMT, payload)
        fw = payload[_INFO_SIZE:_INFO_SIZE + fw_len].decode("ascii", errors="replace")
        return cls(n_classes=n_classes, embed_dim=embed_dim,
                   model_hash=model_hash, firmware=fw)


@dataclass
class Prediction:
    """One prediction, tagged with the EMaGer frame that produced it.

    ``frame_id`` is the base station's own frame counter, copied straight out of
    the incoming frame. It is what makes a prediction verifiable: the same
    capture replayed on the host can be lined up against the device's output
    frame for frame, with no guessing about which window produced what.
    """

    frame_id: int
    seq: int
    class_id: int          # -1 when the device has no prediction (e.g. uncalibrated)
    infer_us: int = 0      # on-device inference time; 0 means "not measured"
    scores: List[float] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.class_id >= 0

    def encode(self) -> bytes:
        return struct.pack(_PRED_FMT, self.frame_id, self.seq, self.class_id,
                           len(self.scores), self.infer_us) + \
            struct.pack(f"<{len(self.scores)}f", *self.scores)

    @classmethod
    def decode(cls, payload: bytes) -> "Prediction":
        if len(payload) < _PRED_SIZE:
            raise ValueError(f"PREDICTION payload too short: {len(payload)} < {_PRED_SIZE}")
        frame_id, seq, class_id, n_scores, infer_us = struct.unpack_from(_PRED_FMT, payload)
        expected = _PRED_SIZE + 4 * n_scores
        if len(payload) < expected:
            raise ValueError(
                f"PREDICTION declares {n_scores} scores ({expected} bytes) "
                f"but payload is {len(payload)}"
            )
        scores = list(struct.unpack_from(f"<{n_scores}f", payload, _PRED_SIZE))
        return cls(frame_id=frame_id, seq=seq, class_id=class_id,
                   infer_us=infer_us, scores=scores)


@dataclass
class Status:
    """Link and device health counters, for diagnosing the relay."""

    frames_ok: int
    frames_bad: int
    resyncs: int
    predictions: int
    uptime_ms: int

    def encode(self) -> bytes:
        return struct.pack(_STATUS_FMT, self.frames_ok, self.frames_bad,
                           self.resyncs, self.predictions, self.uptime_ms)

    @classmethod
    def decode(cls, payload: bytes) -> "Status":
        if len(payload) < _STATUS_SIZE:
            raise ValueError(f"STATUS payload too short: {len(payload)} < {_STATUS_SIZE}")
        return cls(*struct.unpack_from(_STATUS_FMT, payload))


@dataclass
class LogMessage:
    """Free-form device log line.

    Firmware debug output goes through here rather than to a bare ``printf`` so
    it cannot be mistaken for packet bytes and desynchronize the decoder.
    """

    text: str

    def encode(self) -> bytes:
        return self.text.encode("ascii", errors="replace")

    @classmethod
    def decode(cls, payload: bytes) -> "LogMessage":
        return cls(text=payload.decode("ascii", errors="replace"))


Message = Union[Info, Prediction, Status, LogMessage]

_DECODERS = {
    MsgType.INFO: Info,
    MsgType.PREDICTION: Prediction,
    MsgType.STATUS: Status,
    MsgType.LOG: LogMessage,
}


# ---------------------------------------------------------------------------
#  Framing
# ---------------------------------------------------------------------------

def encode_packet(msg_type: Union[MsgType, int], payload: bytes) -> bytes:
    """Wrap a payload in the transport framing.

    Mostly used by tests and by the host-side device simulator; the real encoder
    is the firmware's.
    """
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload of {len(payload)} bytes exceeds MAX_PAYLOAD={MAX_PAYLOAD}")
    body = struct.pack("<BBH", PROTOCOL_VERSION, int(msg_type), len(payload)) + payload
    return SOF + body + struct.pack("<H", crc16(body))


def encode_message(msg: Message) -> bytes:
    """Frame one of the payload dataclasses."""
    for msg_type, kind in _DECODERS.items():
        if isinstance(msg, kind):
            return encode_packet(msg_type, msg.encode())
    raise TypeError(f"not a protocol message: {type(msg).__name__}")


class PacketDecoder:
    """Incremental decoder for the MCU byte stream.

    Feed it whatever bytes arrive; it returns the messages that completed. It
    keeps no timing assumptions, so a packet split across reads is fine, and it
    recovers from garbage by resynchronizing on the start-of-frame marker.
    """

    def __init__(self, max_buffer: int = 8 * (OVERHEAD + MAX_PAYLOAD)):
        self._buf = bytearray()
        self._max_buffer = max_buffer
        # Diagnostics -- a rising resync count usually means the MCU is emitting
        # something that is not a packet, or the link is dropping bytes.
        self.packets_ok = 0
        self.crc_errors = 0
        self.resyncs = 0
        self.dropped_bytes = 0

    def feed(self, data: bytes) -> List[Message]:
        """Consume bytes, return every message that became complete."""
        if data:
            self._buf += data

        if len(self._buf) > self._max_buffer:
            # Nothing valid can be this far back: a packet is at most
            # OVERHEAD + MAX_PAYLOAD bytes, so anything older is unparseable
            # noise. Drop it rather than growing without bound.
            excess = len(self._buf) - self._max_buffer
            del self._buf[:excess]
            self.dropped_bytes += excess

        out: List[Message] = []
        while True:
            msg, consumed, wait_for_more = self._try_one()
            if consumed:
                del self._buf[:consumed]
            if msg is not None:
                out.append(msg)
            if wait_for_more:
                return out

    def _try_one(self):
        """Attempt one packet at the front of the buffer.

        Returns ``(message, bytes_to_consume, wait_for_more)``. ``message`` may
        be None while still consuming bytes -- that is a resync step.
        """
        start = self._buf.find(SOF)
        if start < 0:
            # No marker at all. Keep the last byte: it could be a 0xA5 whose
            # 0x5A has not arrived yet.
            keep = 1 if self._buf[-1:] == SOF[:1] else 0
            drop = len(self._buf) - keep
            self.dropped_bytes += drop
            return None, drop, True

        if start > 0:
            self.dropped_bytes += start
            return None, start, False

        if len(self._buf) < HEADER_SIZE:
            return None, 0, True

        version, msg_type, payload_len = struct.unpack_from("<BBH", self._buf, 2)

        if version != PROTOCOL_VERSION or payload_len > MAX_PAYLOAD:
            # Not a real header -- almost certainly SOF bytes occurring inside
            # payload data. Step past this marker and keep looking.
            self.resyncs += 1
            return None, 1, False

        total = HEADER_SIZE + payload_len + CRC_SIZE
        if len(self._buf) < total:
            return None, 0, True

        body = bytes(self._buf[2:HEADER_SIZE + payload_len])
        (got_crc,) = struct.unpack_from("<H", self._buf, HEADER_SIZE + payload_len)
        if got_crc != crc16(body):
            self.crc_errors += 1
            self.resyncs += 1
            return None, 1, False

        payload = bytes(self._buf[HEADER_SIZE:HEADER_SIZE + payload_len])
        self.packets_ok += 1
        return self._build(msg_type, payload), total, False

    @staticmethod
    def _build(msg_type: int, payload: bytes) -> Optional[Message]:
        """Turn a validated payload into a message, or None if untranslatable.

        A CRC-valid packet of an unknown type is not corruption -- it is newer
        firmware. Log it and move on rather than resynchronizing.
        """
        try:
            kind = MsgType(msg_type)
        except ValueError:
            logger.debug(f"Ignoring unknown message type 0x{msg_type:02X} ({len(payload)} bytes)")
            return None
        try:
            return _DECODERS[kind].decode(payload)
        except (ValueError, struct.error) as exc:
            logger.warning(f"Malformed {kind.name} payload: {exc}")
            return None
