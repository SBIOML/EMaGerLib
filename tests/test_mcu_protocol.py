"""Test suite for the MCU link: wire format and byte-exact relaying.

No hardware: the serial ports are replaced by in-memory fakes, so these run
anywhere.
"""

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from emagerlib.mcu.protocol import (
    MAX_PAYLOAD,
    Info,
    LogMessage,
    MsgType,
    PacketDecoder,
    Prediction,
    Status,
    crc16,
    encode_message,
    encode_packet,
)
from emagerlib.mcu.prediction_reader import PredictionReader
from emagerlib.mcu.relay import FileSource, FrameStats, Relay


def build_frame(counter: int, payload_byte: int = 0x00) -> bytes:
    """A structurally valid EMaGer v3 frame. Contents are irrelevant to the
    relay -- only the header, trailer and counter are ever inspected."""
    frame = bytearray([payload_byte] * 8192)
    frame[0:2] = b"\xAA\x55"
    frame[8176:8180] = counter.to_bytes(4, "big")
    frame[8190:8192] = b"\x55\xAA"
    return bytes(frame)


class FakeSerial:
    """Minimal stand-in for serial.Serial: writes accumulate, reads drain."""

    def __init__(self, to_read: bytes = b""):
        self.written = bytearray()
        self._to_read = bytearray(to_read)
        self.is_open = True

    @property
    def in_waiting(self):
        return len(self._to_read)

    def read(self, n):
        chunk = bytes(self._to_read[:n])
        del self._to_read[:n]
        return chunk

    def write(self, data):
        self.written += data
        return len(data)

    def reset_input_buffer(self):
        self._to_read.clear()

    def close(self):
        self.is_open = False


class FakeSource:
    """Byte source that hands out a fixed list of chunks, then reports exhausted."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.exhausted = False
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def read_chunk(self):
        if self.chunks:
            return self.chunks.pop(0)
        self.exhausted = True
        return b""

    def close(self):
        self.closed = True


class TestCrc(unittest.TestCase):
    """CRC-16/CCITT-FALSE, pinned against its published check value so a
    firmware implementation can be verified against the same constant."""

    def test_01_check_value(self):
        self.assertEqual(crc16(b"123456789"), 0x29B1)

    def test_02_empty_is_init(self):
        self.assertEqual(crc16(b""), 0xFFFF)

    def test_03_detects_single_bit_flip(self):
        data = bytearray(b"emager prediction payload")
        original = crc16(bytes(data))
        data[3] ^= 0x01
        self.assertNotEqual(crc16(bytes(data)), original)


class TestPayloads(unittest.TestCase):
    """Every payload type survives an encode/decode round trip."""

    def test_01_prediction_round_trip(self):
        pred = Prediction(frame_id=123456, seq=42, class_id=3, infer_us=1875,
                          scores=[-0.5, -1.25, -3.0, -0.125])
        got = Prediction.decode(pred.encode())
        self.assertEqual(got.frame_id, 123456)
        self.assertEqual(got.seq, 42)
        self.assertEqual(got.class_id, 3)
        self.assertEqual(got.infer_us, 1875)
        self.assertEqual(got.scores, [-0.5, -1.25, -3.0, -0.125])
        self.assertTrue(got.valid)

    def test_02_prediction_without_scores(self):
        got = Prediction.decode(Prediction(frame_id=1, seq=1, class_id=0).encode())
        self.assertEqual(got.scores, [])

    def test_03_negative_class_is_invalid(self):
        got = Prediction.decode(Prediction(frame_id=1, seq=1, class_id=-1).encode())
        self.assertEqual(got.class_id, -1)
        self.assertFalse(got.valid)

    def test_04_info_round_trip(self):
        info = Info(n_classes=7, embed_dim=64, model_hash=0xDEADBEEF, firmware="emager-hil 0.1")
        got = Info.decode(info.encode())
        self.assertEqual((got.n_classes, got.embed_dim, got.model_hash, got.firmware),
                         (7, 64, 0xDEADBEEF, "emager-hil 0.1"))

    def test_05_status_round_trip(self):
        st = Status(frames_ok=1000, frames_bad=2, resyncs=1, predictions=997, uptime_ms=42000)
        self.assertEqual(Status.decode(st.encode()), st)

    def test_06_log_round_trip(self):
        self.assertEqual(LogMessage.decode(LogMessage("arena used: 15328").encode()).text,
                         "arena used: 15328")

    def test_07_truncated_payload_rejected(self):
        with self.assertRaises(ValueError):
            Prediction.decode(b"\x01\x02\x03")

    def test_08_lying_score_count_rejected(self):
        """A payload claiming more scores than it carries must not be trusted."""
        pred = Prediction(frame_id=1, seq=1, class_id=0, scores=[1.0, 2.0])
        with self.assertRaises(ValueError):
            Prediction.decode(pred.encode()[:-4])


class TestDecoder(unittest.TestCase):

    def test_01_single_packet(self):
        pred = Prediction(frame_id=7, seq=1, class_id=2, scores=[0.0, -1.0])
        msgs = PacketDecoder().feed(encode_message(pred))
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].frame_id, 7)

    def test_02_several_packets_in_one_read(self):
        stream = b"".join(
            encode_message(Prediction(frame_id=i, seq=i, class_id=i % 3)) for i in range(5)
        )
        msgs = PacketDecoder().feed(stream)
        self.assertEqual([m.frame_id for m in msgs], list(range(5)))

    def test_03_packet_split_across_reads(self):
        """Serial reads land on arbitrary boundaries; no split may lose a packet."""
        stream = encode_message(Prediction(frame_id=9, seq=1, class_id=1, scores=[1.5]))
        for split in range(1, len(stream)):
            dec = PacketDecoder()
            msgs = dec.feed(stream[:split]) + dec.feed(stream[split:])
            self.assertEqual(len(msgs), 1, f"lost the packet when split at {split}")
            self.assertEqual(msgs[0].frame_id, 9)

    def test_04_one_byte_at_a_time(self):
        stream = encode_message(Status(1, 2, 3, 4, 5))
        dec = PacketDecoder()
        msgs = []
        for i in range(len(stream)):
            msgs += dec.feed(stream[i:i + 1])
        self.assertEqual(len(msgs), 1)

    def test_05_resyncs_after_leading_garbage(self):
        """Attaching mid-stream, or after boot chatter, must still converge."""
        dec = PacketDecoder()
        msgs = dec.feed(b"MCU booting...\xA5 partial junk \x00\xFF"
                        + encode_message(Prediction(frame_id=3, seq=1, class_id=1)))
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].frame_id, 3)

    def test_06_corrupt_packet_dropped_next_one_survives(self):
        good = encode_message(Prediction(frame_id=1, seq=1, class_id=1))
        corrupt = bytearray(encode_message(Prediction(frame_id=2, seq=2, class_id=2)))
        corrupt[-1] ^= 0xFF          # break the CRC
        dec = PacketDecoder()
        msgs = dec.feed(bytes(corrupt) + good)
        self.assertEqual([m.frame_id for m in msgs], [1])
        self.assertEqual(dec.crc_errors, 1)

    def test_07_sof_inside_payload_does_not_desync(self):
        """0xA5 0x5A can occur in score bytes; the length+CRC must carry us past it."""
        pred = Prediction(frame_id=0xA55AA55A, seq=0xA55AA55A, class_id=1)
        dec = PacketDecoder()
        msgs = dec.feed(encode_message(pred) + encode_message(
            Prediction(frame_id=2, seq=2, class_id=0)))
        self.assertEqual([m.frame_id for m in msgs], [0xA55AA55A, 2])

    def test_08_unknown_type_skipped_without_desync(self):
        """Newer firmware sending a type we don't know must not break the stream."""
        dec = PacketDecoder()
        msgs = dec.feed(encode_packet(0x42, b"future") +
                        encode_message(Prediction(frame_id=5, seq=1, class_id=0)))
        self.assertEqual([m.frame_id for m in msgs], [5])
        self.assertEqual(dec.crc_errors, 0)

    def test_09_oversized_payload_refused_at_encode(self):
        with self.assertRaises(ValueError):
            encode_packet(MsgType.LOG, b"x" * (MAX_PAYLOAD + 1))

    def test_10_pure_garbage_does_not_grow_buffer(self):
        dec = PacketDecoder(max_buffer=1024)
        for _ in range(100):
            dec.feed(b"\xA5" * 512)      # worst case: looks like a SOF start
        self.assertLessEqual(len(dec._buf), 1024)


class TestRelayIsByteExact(unittest.TestCase):
    """The relay's contract: what goes in comes out, unchanged."""

    def test_01_bytes_pass_through_unmodified(self):
        chunks = [build_frame(0), build_frame(1), build_frame(2)]
        relay = Relay(source=FakeSource(chunks), dst_port="fake", stats=False)
        relay.dst = FakeSerial()
        relay.source.open()

        while relay.pump():
            pass

        self.assertEqual(bytes(relay.dst.written), b"".join(chunks))

    def test_02_torn_and_odd_sized_chunks_are_not_realigned(self):
        """Reads split frames arbitrarily; the relay must not buffer to frame
        boundaries, or the MCU would see a different byte timing than the base
        station produces."""
        stream = build_frame(10) + build_frame(11)
        chunks = [stream[:13], stream[13:8000], stream[8000:8195], stream[8195:]]
        relay = Relay(source=FakeSource(chunks), dst_port="fake", stats=False)
        relay.dst = FakeSerial()
        relay.source.open()

        writes = []
        original_write = relay.dst.write

        def spy(data):
            writes.append(bytes(data))
            return original_write(data)

        relay.dst.write = spy
        while relay.pump():
            pass

        self.assertEqual(bytes(relay.dst.written), stream)
        self.assertEqual(writes, chunks, "chunk boundaries were altered")

    def test_03_recording_matches_the_relayed_bytes(self):
        chunks = [build_frame(0), build_frame(1)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.bin"
            relay = Relay(source=FakeSource(chunks), dst_port="fake",
                          record_path=path, stats=False)
            relay.dst = FakeSerial()
            relay.source.open()
            relay._recorder = path.open("wb")
            while relay.pump():
                pass
            relay._recorder.close()

            self.assertEqual(path.read_bytes(), bytes(relay.dst.written))

    def test_04_replayed_capture_is_identical_to_the_original(self):
        original = build_frame(100) + build_frame(101) + build_frame(102)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.bin"
            path.write_bytes(original)

            source = FileSource(path, realtime=False)
            relay = Relay(source=source, dst_port="fake", stats=False)
            relay.dst = FakeSerial()
            source.open()

            stop = threading.Event()
            relay.run(stop)
            source.close()

            self.assertEqual(bytes(relay.dst.written), original)

    def test_05_write_timeout_does_not_kill_the_relay(self):
        import serial as pyserial

        class FlakySerial(FakeSerial):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def write(self, data):
                self.calls += 1
                if self.calls == 1:
                    raise pyserial.SerialTimeoutException("full")
                return super().write(data)

        chunks = [b"first", b"second"]
        relay = Relay(source=FakeSource(chunks), dst_port="fake", stats=False)
        relay.dst = FlakySerial()
        relay.source.open()
        while relay.pump() or not relay.source.exhausted:
            pass

        self.assertEqual(relay.write_timeouts, 1)
        self.assertEqual(bytes(relay.dst.written), b"second")


class TestFrameStats(unittest.TestCase):
    """Passive counters. They observe a copy and must never see the data path."""

    def test_01_counts_valid_frames(self):
        stats = FrameStats()
        stats.feed(b"".join(build_frame(i) for i in range(5)))
        self.assertEqual(stats.frames_ok, 5)
        self.assertEqual(stats.counter_gaps, 0)

    def test_02_detects_a_missing_frame(self):
        stats = FrameStats()
        stats.feed(build_frame(0) + build_frame(1) + build_frame(5))
        self.assertEqual(stats.frames_ok, 3)
        self.assertEqual(stats.counter_gaps, 1)
        self.assertEqual(stats.lost_frames, 3)      # frames 2, 3, 4 never arrived

    def test_03_survives_arbitrary_chunking(self):
        stream = b"".join(build_frame(i) for i in range(4))
        stats = FrameStats()
        for i in range(0, len(stream), 777):
            stats.feed(stream[i:i + 777])
        self.assertEqual(stats.frames_ok, 4)
        self.assertEqual(stats.counter_gaps, 0)

    def test_04_resyncs_after_a_torn_first_frame(self):
        """Attaching mid-frame is the normal startup case, not an error."""
        stream = build_frame(0)[4000:] + build_frame(1) + build_frame(2)
        stats = FrameStats()
        stats.feed(stream)
        self.assertEqual(stats.frames_ok, 2)


class TestPredictionReader(unittest.TestCase):
    """The reader thread, driven by a fake port -- no hardware, no MCU."""

    def _drain(self, ser, **kwargs):
        """Run a reader over a preloaded fake port until it stops producing."""
        received = []
        done = threading.Event()

        def on_prediction(pred):
            received.append(pred)
            done.set()

        reader = PredictionReader(ser, on_prediction=on_prediction, **kwargs)
        reader.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and ser.in_waiting:
            time.sleep(0.005)
        time.sleep(0.05)          # let the last packet through the decoder
        reader.stop()
        return reader, received

    def test_01_predictions_reach_the_callback(self):
        stream = b"".join(
            encode_message(Prediction(frame_id=100 + i, seq=i, class_id=i % 3))
            for i in range(4)
        )
        reader, received = self._drain(FakeSerial(stream))
        self.assertEqual([p.frame_id for p in received], [100, 101, 102, 103])
        self.assertEqual(reader.n_predictions, 4)
        self.assertEqual(reader.seq_gaps, 0)

    def test_02_latest_holds_the_most_recent(self):
        stream = b"".join(
            encode_message(Prediction(frame_id=i, seq=i, class_id=1)) for i in range(3)
        )
        reader, _ = self._drain(FakeSerial(stream))
        self.assertEqual(reader.latest.frame_id, 2)

    def test_03_detects_dropped_predictions(self):
        stream = (encode_message(Prediction(frame_id=1, seq=1, class_id=0))
                  + encode_message(Prediction(frame_id=2, seq=5, class_id=0)))
        reader, _ = self._drain(FakeSerial(stream))
        self.assertEqual(reader.seq_gaps, 1)
        self.assertEqual(reader.lost_predictions, 3)   # seq 2, 3, 4

    def test_04_info_and_status_are_captured(self):
        stream = (encode_message(Info(7, 64, 0xABCDEF01, "fw-test"))
                  + encode_message(Status(10, 0, 0, 9, 1234))
                  + encode_message(Prediction(frame_id=1, seq=1, class_id=0)))
        reader, _ = self._drain(FakeSerial(stream))
        self.assertEqual(reader.info.n_classes, 7)
        self.assertEqual(reader.info.firmware, "fw-test")
        self.assertEqual(reader.status.predictions, 9)

    def test_05_boot_chatter_before_the_first_packet_is_tolerated(self):
        stream = (b"\r\nMCU booting, arena used: 15328\r\n"
                  + encode_message(Prediction(frame_id=42, seq=1, class_id=2)))
        reader, received = self._drain(FakeSerial(stream))
        self.assertEqual([p.frame_id for p in received], [42])

    def test_06_a_failing_callback_does_not_stop_the_reader(self):
        stream = b"".join(
            encode_message(Prediction(frame_id=i, seq=i, class_id=0)) for i in range(3)
        )
        seen = []

        def boom(pred):
            seen.append(pred.frame_id)
            raise RuntimeError("display blew up")

        reader = PredictionReader(FakeSerial(stream), on_prediction=boom)
        reader.start()
        time.sleep(0.1)
        reader.stop()
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(reader.n_predictions, 3)


if __name__ == "__main__":
    unittest.main()
