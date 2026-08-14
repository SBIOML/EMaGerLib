# MCU → PC prediction protocol

Specification of the packets the microcontroller sends back to the PC when it is
running the gesture model. **This document is the contract**: the host side is
implemented in this repository (`emagerlib/mcu/`), the device side belongs in the
firmware repository. Implement this and the two halves meet.

```
EMaGer v3  ──WiFi──►  base station  ──USB──►  PC  ──USB──►  MCU
                                               ▲               │
                                               └───────────────┘
                                                  predictions
```

The PC is a wire in the downstream direction: it forwards the base station's raw
bytes **verbatim**, so the MCU sees exactly the stream it would see if the base
station were wired to it directly. Nothing is filtered, reframed, or
re-timestamped. Everything from frame decoding to inference happens in the
firmware.

---

## Contents

- [Design constraints](#design-constraints)
- [Packet layout](#packet-layout)
- [CRC](#crc)
- [Message types](#message-types)
  - [0x02 PREDICTION](#0x02-prediction)
  - [0x01 INFO](#0x01-info)
  - [0x03 STATUS](#0x03-status)
  - [0x7F LOG](#0x7f-log)
- [How the host resynchronizes](#how-the-host-resynchronizes)
- [Reference C implementation](#reference-c-implementation)
- [Test vectors](#test-vectors)
- [Implementation checklist](#implementation-checklist)

---

## Design constraints

**The link is one-way: the MCU talks, the PC listens.** There is no
host-to-device control channel, and this is deliberate. The PC→MCU direction
carries the base station's byte stream and nothing else — injecting control
packets into it would make the stream differ from what the MCU receives when the
base station is wired directly, which is the one property the whole setup exists
to preserve. Anything the host would otherwise ask for, the MCU volunteers (see
[INFO](#0x01-info)).

**Little-endian throughout.** Both ends are little-endian, so neither byte-swaps.
Note this differs from the EMaGer frame counter, which is **big-endian** on the
wire: byte-swap it once when reading the incoming frame, then keep everything
native from there.

**Self-delimiting and resynchronizable.** The host may attach mid-stream, after
the MCU has already booted and printed things. Start marker + length + CRC lets
it converge on the next valid packet without any handshake.

**Throughput is a non-issue upstream.** A prediction packet is 20–48 bytes at
roughly 24 Hz — about 1 kB/s. The *downstream* direction is the demanding one
(~195 kB/s: 8192-byte frames at 23.8 frames/s), so a real UART needs at least
2 Mbaud; over USB CDC the baud rate is nominal and anything works.

---

## Packet layout

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | Start of frame | `0xA5 0x5A`, in this order |
| 2 | 1 | Protocol version | `0x01` |
| 3 | 1 | Message type | see below |
| 4 | 2 | Payload length | `uint16` LE, ≤ 512 |
| 6 | N | Payload | type-specific |
| 6+N | 2 | CRC-16 | `uint16` LE, over bytes `[2 .. 6+N-1]` |

Total size is `8 + N`. The CRC covers **version, type, length and payload** — not
the start marker, which is a search pattern rather than data.

Max payload is **512 bytes**; the host refuses anything larger, treating an
oversized length field as evidence it is not looking at a real header. The
biggest real packet is a PREDICTION with one score per class, well under that.

---

## CRC

**CRC-16/CCITT-FALSE**: polynomial `0x1021`, initial value `0xFFFF`, no input or
output reflection, no final XOR.

Check value: `crc16("123456789") == 0x29B1`. Pin a unit test to that constant on
the firmware side — it catches the reflection and init-value mistakes that make
two "CRC-16" implementations disagree.

---

## Message types

| Type | Name | Direction | When |
|---|---|---|---|
| `0x01` | INFO | MCU → PC | at boot, then periodically |
| `0x02` | PREDICTION | MCU → PC | once per inference |
| `0x03` | STATUS | MCU → PC | periodically (e.g. every 1 s) |
| `0x7F` | LOG | MCU → PC | free-form device logging |

A CRC-valid packet of an unrecognized type is **skipped, not treated as
corruption** — that keeps newer firmware from breaking an older host. You may add
types without coordinating a host release.

### 0x02 PREDICTION

Payload: 12 bytes + 4 bytes per score.

| Offset | Type | Field | Notes |
|---|---|---|---|
| 0 | `uint32` | `frame_id` | EMaGer frame counter of the frame that produced this prediction |
| 4 | `uint32` | `seq` | increments by 1 per prediction; lets the host detect loss |
| 8 | `int8` | `class_id` | predicted class, or `-1` if there is none |
| 9 | `uint8` | `n_scores` | number of float32 scores that follow; `0` is valid |
| 10 | `uint16` | `infer_us` | on-device inference time in µs; `0` = not measured |
| 12 | `float32 × n_scores` | `scores` | per-class score, class order = training order |

**`frame_id` is the field that makes a prediction verifiable.** Copy it straight
out of the frame you consumed (remembering it is big-endian in the frame). It
lets the host replay the identical capture and line up device predictions with
host-computed ones frame for frame, instead of guessing which window produced
what. If a prediction consumes several frames, report the **last** one that went
into it.

`class_id = -1` is the honest answer when the device cannot classify — for the
prototypical models, that is `emager_classify()` returning `-1` because no
prototype is valid. Do not substitute class 0.

`scores` are optional. When present, use the same quantity the PyTorch model
produces, `score[c] = -‖e − p_c‖²/D`, so a softmax over them matches
`predict_proba()`. Sending none is fine and costs nothing at the host.

### 0x01 INFO

Payload: 8 bytes + firmware name.

| Offset | Type | Field |
|---|---|---|
| 0 | `uint8` | `n_classes` |
| 1 | `uint16` | `embed_dim` |
| 3 | `uint32` | `model_hash` |
| 7 | `uint8` | `fw_len` |
| 8 | `char × fw_len` | firmware name/version, ASCII, not NUL-terminated |

Emit at boot **and periodically** (every few seconds is plenty). Because there is
no host-to-device channel, a PC that attaches after boot has no way to ask — so
the MCU has to keep offering.

`model_hash` should identify the exact network in flash (e.g. a CRC or truncated
SHA of `emager_model_data.h`). This is the guard against the failure mode called
out in [`examples/deployment/README.md`](../examples/deployment/README.md):
prototypes paired with weights they were not computed from, which fails silently
as wrong predictions rather than as an error.

### 0x03 STATUS

Payload: 20 bytes, five `uint32`.

| Offset | Field | Meaning |
|---|---|---|
| 0 | `frames_ok` | EMaGer frames decoded with a valid trailer |
| 4 | `frames_bad` | frames rejected (bad trailer) |
| 8 | `resyncs` | times the frame parser had to hunt for the header |
| 12 | `predictions` | inferences run |
| 16 | `uptime_ms` | milliseconds since boot |

This is how a link problem gets attributed. The host counts the same things on
its side of the relay, so comparing the two says whether bytes are being lost
between the PC and the MCU or upstream of the PC.

### 0x7F LOG

Payload: ASCII text, no NUL terminator, ≤ 512 bytes.

Route firmware debug output through this rather than writing bare text to the
same port. Unframed `printf` output is indistinguishable from packet bytes and
will make the host's decoder resynchronize, which shows up as mysterious
intermittent packet loss.

---

## How the host resynchronizes

Useful to know, because it defines what the firmware can get away with:

1. Scan for `0xA5 0x5A`.
2. Check the version byte is `0x01` and the length is ≤ 512. If not, advance one
   byte and rescan.
3. Wait for the full packet if it has not all arrived.
4. Verify the CRC. If it fails, advance one byte from the start marker and
   rescan.
5. Dispatch on the type; unknown types are skipped.

A packet split across any number of reads is reassembled correctly (there is a
test for every possible split point). Bytes that are not part of a valid packet
are discarded silently. The one thing that costs you is *volume* of junk — the
host counts `resyncs`, so a rising count means the device is emitting something
that is not a packet.

---

## Reference C implementation

Transmit side, freestanding C99 — no allocation, nothing beyond `<string.h>`.

```c
#include <stdint.h>
#include <string.h>

#define EMAGER_PKT_SOF0     0xA5
#define EMAGER_PKT_SOF1     0x5A
#define EMAGER_PKT_VERSION  0x01

#define EMAGER_MSG_INFO        0x01
#define EMAGER_MSG_PREDICTION  0x02
#define EMAGER_MSG_STATUS      0x03
#define EMAGER_MSG_LOG         0x7F

/* CRC-16/CCITT-FALSE. crc16("123456789") == 0x29B1 */
static uint16_t emager_crc16(const uint8_t *data, uint32_t len)
{
    uint16_t crc = 0xFFFF;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                                 : (uint16_t)(crc << 1);
    }
    return crc;
}

/* Frame a payload into `out` (needs payload_len + 8 bytes). Returns total size. */
static uint32_t emager_pkt_build(uint8_t *out, uint8_t type,
                                 const uint8_t *payload, uint16_t payload_len)
{
    out[0] = EMAGER_PKT_SOF0;
    out[1] = EMAGER_PKT_SOF1;
    out[2] = EMAGER_PKT_VERSION;
    out[3] = type;
    out[4] = (uint8_t)(payload_len & 0xFF);
    out[5] = (uint8_t)(payload_len >> 8);
    if (payload_len)
        memcpy(out + 6, payload, payload_len);

    /* CRC covers version..payload, i.e. everything after the start marker. */
    uint16_t crc = emager_crc16(out + 2, (uint32_t)payload_len + 4);
    out[6 + payload_len] = (uint8_t)(crc & 0xFF);
    out[7 + payload_len] = (uint8_t)(crc >> 8);
    return (uint32_t)payload_len + 8;
}

/* --- little-endian writers; explicit so packing/alignment cannot bite --- */
static uint32_t put_u8 (uint8_t *p, uint32_t o, uint8_t v)  { p[o] = v; return o + 1; }
static uint32_t put_u16(uint8_t *p, uint32_t o, uint16_t v) { p[o]=(uint8_t)v; p[o+1]=(uint8_t)(v>>8); return o+2; }
static uint32_t put_u32(uint8_t *p, uint32_t o, uint32_t v)
{
    p[o]=(uint8_t)v; p[o+1]=(uint8_t)(v>>8); p[o+2]=(uint8_t)(v>>16); p[o+3]=(uint8_t)(v>>24);
    return o + 4;
}
static uint32_t put_f32(uint8_t *p, uint32_t o, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, 4);          /* not a cast: avoids the aliasing trap */
    return put_u32(p, o, bits);
}

/*
 * Send one prediction.
 *
 * frame_id : the EMaGer frame counter this prediction came from. It is
 *            BIG-endian in the incoming frame at offset 8176 -- swap it once
 *            when you read the frame, and pass the native value here.
 * class_id : predicted class, or -1 if there is none (e.g. emager_classify()
 *            returned -1 because no prototype is valid). Do NOT send 0 instead.
 * scores   : may be NULL with n_scores = 0.
 */
void emager_send_prediction(uint32_t frame_id, uint32_t seq, int8_t class_id,
                            uint16_t infer_us, const float *scores, uint8_t n_scores)
{
    uint8_t payload[12 + 4 * 32];
    uint8_t packet[sizeof(payload) + 8];

    uint32_t o = 0;
    o = put_u32(payload, o, frame_id);
    o = put_u32(payload, o, seq);
    o = put_u8 (payload, o, (uint8_t)class_id);   /* two's complement: -1 -> 0xFF */
    o = put_u8 (payload, o, n_scores);
    o = put_u16(payload, o, infer_us);
    for (uint8_t i = 0; i < n_scores; i++)
        o = put_f32(payload, o, scores[i]);

    uint32_t n = emager_pkt_build(packet, EMAGER_MSG_PREDICTION, payload, (uint16_t)o);
    uart_write(packet, n);        /* your transport */
}
```

`float32` goes on the wire as raw IEEE-754 little-endian bits, which is already
the in-memory representation on every target this is likely to run on — the
`memcpy` in `put_f32` is there to dodge strict-aliasing, not to convert.

---

## Test vectors

Byte-for-byte output of the host implementation. If the firmware produces these
exact bytes for these inputs, the two sides agree — including the CRC, the field
order, and the endianness.

**PREDICTION** — `frame_id=1000, seq=7, class_id=3, infer_us=1875`, no scores:

```
A5 5A 01 02 0C 00 E8 03 00 00 07 00 00 00 03 00 53 07 E5 94        (20 bytes)
```

**PREDICTION** — same, with `scores = [-0.5, -1.25, -3.0]`:

```
A5 5A 01 02 18 00 E8 03 00 00 07 00 00 00 03 03 53 07
00 00 00 BF 00 00 A0 BF 00 00 40 C0 1C C6                          (32 bytes)
```

**PREDICTION** — `frame_id=1, seq=1, class_id=-1` (nothing to report):

```
A5 5A 01 02 0C 00 01 00 00 00 01 00 00 00 FF 00 00 00 3C 96        (20 bytes)
```

**INFO** — `n_classes=7, embed_dim=64, model_hash=0xDEADBEEF, "emager-hil 0.1"`:

```
A5 5A 01 01 16 00 07 40 00 EF BE AD DE 0E 65 6D 61 67 65 72
2D 68 69 6C 20 30 2E 31 3E D0                                      (30 bytes)
```

**STATUS** — `frames_ok=1000, frames_bad=2, resyncs=1, predictions=997, uptime_ms=42000`:

```
A5 5A 01 03 14 00 E8 03 00 00 02 00 00 00 01 00 00 00 E5 03
00 00 10 A4 00 00 F8 7A                                            (28 bytes)
```

**LOG** — `"arena used: 15328"`:

```
A5 5A 01 7F 11 00 61 72 65 6E 61 20 75 73 65 64 3A 20 31 35
33 32 38 D5 FC                                                     (25 bytes)
```

Regenerate any of these with:

```python
from emagerlib.mcu.protocol import Prediction, encode_message
print(encode_message(Prediction(frame_id=1000, seq=7, class_id=3, infer_us=1875)).hex(" "))
```

---

## Implementation checklist

- [ ] `crc16("123456789") == 0x29B1` passes as a unit test.
- [ ] The test vectors above reproduce byte for byte.
- [ ] `frame_id` is byte-swapped from the incoming frame (big-endian → native).
- [ ] `seq` increments by exactly 1 per prediction, and wraps at 2³².
- [ ] `class_id = -1` is sent when there is no prediction, not class 0.
- [ ] INFO is emitted at boot **and** periodically.
- [ ] All debug output goes through LOG — no bare `printf` on the packet port.
- [ ] `model_hash` changes when the network in flash changes.

To test without an armband, record a session on the PC and replay it:

```bash
emager mcu-predict --mcu-port <port> --record capture.bin   # once, with hardware
emager mcu-predict --mcu-port <port> --replay capture.bin   # thereafter
```

The replayed bytes are identical to the live ones, so the MCU cannot tell the
difference — which makes a firmware bug reproducible instead of requiring the
armband on someone's arm.
