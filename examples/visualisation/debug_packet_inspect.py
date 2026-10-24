#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grab ONE raw frame straight from the serial port (bypassing libemg's
shared-memory streamer entirely) and print it out index-by-index so you can
verify the wire format matches what the streamer expects.

Supports both frame formats (--format v3 | puck):

  Emager v3.0 (used by emagerv3_streamer / live_64_channel_v3.py), 8192 bytes:
    [0]          0xAA
    [1]          0x55
    [2:8066]     EMG payload (8064 bytes) = 5376 packed 12-bit values (84 samples x 64 ch)
    [8066:8174]  IMU payload area (108 bytes max)
    [8174]       IMU sample count (expected 8 or 9)
    [8175]       Alignment IMU (ignored)
    [8176:8180]  Counter, big-endian uint32
    [8180:8190]  Empty (ignored)
    [8190]       0x55
    [8191]       0xAA

  EmagerPuck (used by emagerPuck_streamer), 8192 bytes:
    [0]          0xAA
    [1]          0x55
    [2:6]        frame counter, big-endian uint32
    [6:8166]     packed 15-bit signed EMG payload (8160 bytes -> 32 ch)
    [8190]       0x55
    [8191]       0xAA

Usage:
  python debug_packet_inspect.py --format v3 [--baud 3000000] [--com-name COM5] [--dump-hex]
  python debug_packet_inspect.py --format puck ...
"""

import argparse
import platform
import time

import numpy as np
import serial
import serial.tools.list_ports

# Reuse the real parsing/unpacking logic from libemg so this debug view can
# never silently drift out of sync with what the actual streamer does.
from libemg._streamers._emagerv3_streamer import Emager3
from libemg._streamers._emagerPuck_streamer import EmagerPuck

FORMATS = {"v3": Emager3, "puck": EmagerPuck}


def find_port(com_name, vid_pid):
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if com_name is None:
            if (p.vid, p.pid) == vid_pid:
                return p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")
        else:
            dev = getattr(p, "device", "") or ""
            name = getattr(p, "name", "") or ""
            desc = getattr(p, "description", "") or ""
            if com_name in dev or com_name in name or com_name in desc:
                return p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")

    avail = "\n".join(
        f"  - {getattr(p, 'device', '?')} - {getattr(p, 'description', '')} "
        f"(VID: {getattr(p, 'vid', None)}, PID: {getattr(p, 'pid', None)})"
        for p in ports
    ) or "  (no serial ports found)"
    raise RuntimeError(f"Could not find serial port. Available ports:\n{avail}")


def capture_one_frame(ser, spec):
    """Read raw bytes until one full, header/trailer-aligned frame is found."""
    hdr = bytes([spec.HDR0, spec.HDR1])
    frame_size = spec.FRAME_SIZE
    buf = bytearray()
    pos = 0

    while True:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk

        h = buf.find(hdr, pos)
        if h < 0:
            pos = max(0, len(buf) - 1)
            continue

        if len(buf) - h < frame_size:
            pos = h
            continue

        t0, t1 = buf[h + spec.TRAILER0_I], buf[h + spec.TRAILER1_I]
        if t0 == spec.TLR0 and t1 == spec.TLR1:
            return bytes(buf[h:h + frame_size])

        # bad trailer -> resync one byte forward and keep scanning
        pos = h + 1


def run_live_monitor(ser, duration_s):
    """
    Mirrors Emager3.get_data()'s parse loop, but NEVER lets a decode exception
    kill the loop -- it reports it instead. This is meant to catch, without
    crashing, exactly the kind of malformed frame that would silently kill
    the real EmagerV3Streamer background process (its main loop calls
    self.e.get_data() with no try/except, so one bad frame there stops all
    data for the rest of the run).

    Prints a status line once per second: how many frames parsed cleanly,
    how many had a bad trailer (resync), how many counter gaps, how many
    frames raised a decode exception, the running set of distinct IMU
    sample-count byte values seen (should only ever be 8 or 9), and the
    last frame counter value (freezing counter == device/stream stalled).
    """
    spec = Emager3
    hdr = bytes([spec.HDR0, spec.HDR1])
    buf = bytearray()
    pos = 0

    stats = {"frames_ok": 0, "bad_tlr": 0, "ctr_miss": 0, "exceptions": 0}
    last_ctr = None
    imu_nsamp_seen = set()

    t_start = time.monotonic()
    t_last_report = t_start
    last_frames_ok_at_report = 0

    print(f"[INFO] Live monitor running for {duration_s:.0f}s (Ctrl+C to stop early)...\n")

    try:
        while time.monotonic() - t_start < duration_s:
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
            else:
                time.sleep(0.001)

            while True:
                h = buf.find(hdr, pos)
                if h < 0:
                    keep = min(len(buf), spec.FRAME_SIZE - 1)
                    buf = buf[-keep:] if keep else bytearray()
                    pos = 0
                    break

                if len(buf) - h < spec.FRAME_SIZE:
                    if h > 0:
                        buf = buf[h:]
                        pos = 0
                    else:
                        pos = h
                    break

                t0, t1 = buf[h + spec.TRAILER0_I], buf[h + spec.TRAILER1_I]
                if t0 != spec.TLR0 or t1 != spec.TLR1:
                    stats["bad_tlr"] += 1
                    pos = h + 1
                    if pos > spec.FRAME_SIZE * 2:
                        buf = buf[pos:]
                        pos = 0
                    continue

                stats["frames_ok"] += 1
                counter = None
                try:
                    c0, c1, c2, c3 = (buf[h + spec.CTR_START + i] for i in range(4))
                    counter = (c0 << 24) | (c1 << 16) | (c2 << 8) | c3
                    if last_ctr is not None and counter != ((last_ctr + 1) & 0xFFFFFFFF):
                        stats["ctr_miss"] += 1
                    last_ctr = counter

                    imu_nsamp = int(buf[h + spec.IMU_NSAMPLES_I])
                    imu_nsamp_seen.add(imu_nsamp)
                    imu_bytes_used = imu_nsamp * spec.IMU_BYTES_PER_SAMPLE

                    emg_bytes = bytes(buf[h + spec.EMG_START: h + spec.EMG_END])
                    imu_bytes = bytes(buf[h + spec.IMU_START: h + spec.IMU_START + imu_bytes_used])

                    e = Emager3.__new__(Emager3)
                    emg_vals = e._unpack_12bit_be(emg_bytes, n_values=spec.EMG_VALUES_PER_FRAME)
                    emg_vals.reshape(spec.SAMPLES_PER_CH_PER_FRAME, spec.CHANNELS)  # raises if malformed

                    imu_arr = np.frombuffer(imu_bytes, dtype=np.dtype(">i2"))
                    imu_arr.reshape(imu_nsamp, spec.IMU_AXES)  # raises if imu_nsamp is garbage

                except Exception as exc:
                    stats["exceptions"] += 1
                    print(f"[EXCEPTION #{stats['exceptions']}] at t={time.monotonic() - t_start:.2f}s, "
                          f"frame #{stats['frames_ok']}: {type(exc).__name__}: {exc}")
                    print(f"    counter={counter}  imu_nsamp_byte={buf[h + spec.IMU_NSAMPLES_I]}  "
                          f"-> THIS is the kind of frame that would silently kill the real streamer process.")

                pos = h + spec.FRAME_SIZE
                if pos > spec.FRAME_SIZE * 2:
                    buf = buf[pos:]
                    pos = 0

            now = time.monotonic()
            if now - t_last_report >= 1.0:
                new_frames = stats["frames_ok"] - last_frames_ok_at_report
                print(f"[{now - t_start:6.1f}s] +{new_frames:4d} frames/s | "
                      f"total_ok={stats['frames_ok']} bad_tlr={stats['bad_tlr']} "
                      f"ctr_miss={stats['ctr_miss']} exceptions={stats['exceptions']} "
                      f"last_ctr={last_ctr} imu_nsamp_seen={sorted(imu_nsamp_seen)}")
                last_frames_ok_at_report = stats["frames_ok"]
                t_last_report = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    print("\n[SUMMARY]")
    print(f"  frames_ok  = {stats['frames_ok']}")
    print(f"  bad_tlr    = {stats['bad_tlr']}  (resync events)")
    print(f"  ctr_miss   = {stats['ctr_miss']}  (dropped/out-of-order frames)")
    print(f"  exceptions = {stats['exceptions']}  (frames that would have crashed the real streamer)")
    print(f"  imu_nsamp values seen = {sorted(imu_nsamp_seen)} (expected only {{8, 9}})")


def inspect_v3(frame, dump_hex):
    spec = Emager3

    print(f"[0]  header byte 0 = 0x{frame[0]:02X} (expected 0xAA)")
    print(f"[1]  header byte 1 = 0x{frame[1]:02X} (expected 0x55)")
    print(f"[{spec.TRAILER0_I}] trailer byte 0 = 0x{frame[spec.TRAILER0_I]:02X} (expected 0x55)")
    print(f"[{spec.TRAILER1_I}] trailer byte 1 = 0x{frame[spec.TRAILER1_I]:02X} (expected 0xAA)")

    c0, c1, c2, c3 = (frame[spec.CTR_START + i] for i in range(4))
    counter = (c0 << 24) | (c1 << 16) | (c2 << 8) | c3
    print(f"\n[{spec.CTR_START}:{spec.CTR_START + 4}] counter bytes = {c0:02X} {c1:02X} {c2:02X} {c3:02X}"
          f"  -> uint32 (big-endian) = {counter}")

    imu_nsamp = int(frame[spec.IMU_NSAMPLES_I])
    imu_bytes_used = imu_nsamp * spec.IMU_BYTES_PER_SAMPLE
    imu_bytes = frame[spec.IMU_START:spec.IMU_START + imu_bytes_used]
    print(f"\n[{spec.IMU_NSAMPLES_I}] IMU sample count = {imu_nsamp} (expected 8 or 9)")
    print(f"[{spec.IMU_START}:{spec.IMU_START + imu_bytes_used}] IMU payload used = {imu_bytes_used} bytes"
          f" (area allows up to {spec.IMU_AREA_LEN} bytes)")
    print(f"  raw bytes: {' '.join(f'{b:02X}' for b in imu_bytes)}")

    emg_bytes = frame[spec.EMG_START:spec.EMG_END]
    print(f"\n[{spec.EMG_START}:{spec.EMG_END}] EMG payload = {len(emg_bytes)} bytes (expected {spec.EMG_LEN})")
    print(f"  first 16 bytes: {' '.join(f'{b:02X}' for b in emg_bytes[:16])}")
    print(f"  last  16 bytes: {' '.join(f'{b:02X}' for b in emg_bytes[-16:])}")

    if dump_hex:
        _dump_hex(frame)

    e = Emager3.__new__(Emager3)
    emg_vals = e._unpack_12bit_be(emg_bytes, n_values=spec.EMG_VALUES_PER_FRAME)
    print(f"\n[DECODED EMG] {emg_vals.size} / {spec.EMG_VALUES_PER_FRAME} 12-bit values unpacked")

    if emg_vals.size == spec.EMG_VALUES_PER_FRAME:
        emg_block = emg_vals.reshape(spec.SAMPLES_PER_CH_PER_FRAME, spec.CHANNELS)
        print(f"  reshaped to (samples={spec.SAMPLES_PER_CH_PER_FRAME}, channels={spec.CHANNELS})")
        print(f"  min={emg_block.min()}  max={emg_block.max()}  mean={emg_block.mean():.1f}")
        print("\n  first 3 samples x 64 channels:")
        np.set_printoptions(linewidth=220)
        print(emg_block[:3])

    if imu_nsamp > 0:
        imu_dtype = np.dtype(">i2")
        imu_arr = np.frombuffer(imu_bytes, dtype=imu_dtype)
        print(f"\n[DECODED IMU] {imu_arr.size} / {imu_nsamp * spec.IMU_AXES} int16 values unpacked")

        if imu_arr.size == imu_nsamp * spec.IMU_AXES:
            imu_block = imu_arr.reshape(imu_nsamp, spec.IMU_AXES)
            axis_names = ["ax", "ay", "az", "gx", "gy", "gz"]
            header = "  sample  " + "  ".join(f"{n:>7}" for n in axis_names)
            print(f"  reshaped to (samples={imu_nsamp}, axes={spec.IMU_AXES})")
            print(header)
            for i, row in enumerate(imu_block):
                vals = "  ".join(f"{v:7d}" for v in row)
                print(f"  {i:6d}  {vals}")
        else:
            print("  WARNING: byte count did not divide evenly into "
                  f"{spec.IMU_AXES} int16 axes; raw values: {list(imu_arr)}")
    else:
        print("\n[DECODED IMU] IMU sample count is 0 -- no IMU values in this frame")


def inspect_puck(frame, dump_hex):
    spec = EmagerPuck

    print(f"[0]  header byte 0 = 0x{frame[0]:02X} (expected 0xAA)")
    print(f"[1]  header byte 1 = 0x{frame[1]:02X} (expected 0x55)")
    print(f"[{spec.TRAILER0_I}] trailer byte 0 = 0x{frame[spec.TRAILER0_I]:02X} (expected 0x55)")
    print(f"[{spec.TRAILER1_I}] trailer byte 1 = 0x{frame[spec.TRAILER1_I]:02X} (expected 0xAA)")

    c0, c1, c2, c3 = (frame[spec.CTR_START + i] for i in range(4))
    counter = (c0 << 24) | (c1 << 16) | (c2 << 8) | c3
    print(f"\n[{spec.CTR_START}:{spec.CTR_START + 4}] counter bytes = {c0:02X} {c1:02X} {c2:02X} {c3:02X}"
          f"  -> uint32 (big-endian) = {counter}")

    emg_bytes = frame[spec.EMG_START:spec.EMG_END]
    print(f"\n[{spec.EMG_START}:{spec.EMG_END}] EMG payload = {len(emg_bytes)} bytes (expected {spec.EMG_LEN})")
    print(f"  first 16 bytes: {' '.join(f'{b:02X}' for b in emg_bytes[:16])}")
    print(f"  last  16 bytes: {' '.join(f'{b:02X}' for b in emg_bytes[-16:])}")

    if dump_hex:
        _dump_hex(frame)

    puck = EmagerPuck.__new__(EmagerPuck)
    puck.restore_lsb0 = True
    emg_vals = puck._unpack_15bit_msbfirst_signed(emg_bytes, n_values=spec.EMG_VALUES_PER_FRAME)
    print(f"\n[DECODED EMG] {emg_vals.size} / {spec.EMG_VALUES_PER_FRAME} 15-bit values unpacked")

    if emg_vals.size == spec.EMG_VALUES_PER_FRAME:
        emg_block = emg_vals.reshape(spec.SAMPLES_PER_CH_PER_FRAME, spec.CHANNELS)
        emg_block = emg_block[:, spec.CHANNEL_ORDER]
        print(f"  reshaped to (samples={spec.SAMPLES_PER_CH_PER_FRAME}, channels={spec.CHANNELS})")
        print(f"  min={emg_block.min()}  max={emg_block.max()}  mean={emg_block.mean():.1f}")
        print("\n  first 3 samples x 32 channels (post channel-reorder):")
        np.set_printoptions(linewidth=200)
        print(emg_block[:3])


def _dump_hex(frame):
    print("\n[FULL FRAME HEX DUMP]")
    for i in range(0, len(frame), 16):
        row = frame[i:i + 16]
        hex_str = " ".join(f"{b:02X}" for b in row)
        print(f"{i:5d}: {hex_str}")


def main():
    ap = argparse.ArgumentParser(description="Dump one raw Emager frame from the serial port.")
    ap.add_argument("--format", choices=list(FORMATS.keys()), default="v3",
                    help="Frame format to decode (default: v3)")
    ap.add_argument("--baud", type=int, default=3000000)
    ap.add_argument("--com-name", type=str, default=None, help="Substring to match a COM port (e.g. 'COM5')")
    ap.add_argument("--vid", type=int, default=12259)
    ap.add_argument("--pid", type=int, default=256)
    ap.add_argument("--dump-hex", action="store_true", help="Print every byte of the frame as hex")
    ap.add_argument("--live", action="store_true",
                    help="Instead of dumping one frame, continuously parse frames (v3 only) and report "
                         "resync/counter-gap/decode-exception stats -- run this during an actual "
                         "repetition to catch a malformed frame in the act.")
    ap.add_argument("--duration", type=float, default=30.0, help="Duration in seconds for --live mode")
    args = ap.parse_args()

    spec = FORMATS[args.format]

    com_port = find_port(args.com_name, (args.vid, args.pid))
    print(f"[INFO] Opening {com_port} @ {args.baud} baud (format={args.format})")

    ser = serial.Serial(com_port, args.baud, timeout=0.1)
    try:
        ser.reset_input_buffer()

        if args.live:
            if args.format != "v3":
                raise SystemExit("--live is only implemented for --format v3")
            run_live_monitor(ser, args.duration)
            return

        print("[INFO] Waiting for one full, valid frame...")
        frame = capture_one_frame(ser, spec)
    finally:
        ser.close()

    print(f"\n[OK] Captured frame: {len(frame)} bytes\n")

    if args.format == "v3":
        inspect_v3(frame, args.dump_hex)
    else:
        inspect_puck(frame, args.dump_hex)


if __name__ == "__main__":
    main()
