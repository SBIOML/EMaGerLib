# Microcontroller deployment

Convert a trained EMaGer model into microcontroller-ready artifacts, verify at every
stage that the converted model still matches the PyTorch original, and — for the
prototypical models — ship the C runtime that lets the **device recalibrate itself to
its user without any training on the device**.

```
PyTorch  ->  ONNX  ->  TFLite float32  ->  TFLite full-INT8  ->  C array
             |             |                    |                   |
        model.onnx  model_float32.tflite  model_int8.tflite  emager_model_data.h
```

Every exported model takes a single **`(1, 64)` float32 MAV feature vector**
(mean-absolute-value over a 200-sample window on the 4×16 electrode grid). Batch size
is fixed to 1 — one inference at a time, the way an MCU runs it.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Which model should I export?](#which-model-should-i-export)
- [Prototypical models: why the graph stops at the embedding](#prototypical-models-why-the-graph-stops-at-the-embedding)
- [Output artifacts](#output-artifacts)
- [On the MCU](#on-the-mcu)
  - [Memory budget](#memory-budget)
  - [1. Drop the files in](#1-drop-the-files-in)
  - [2. Set up TFLite Micro](#2-set-up-tflite-micro)
  - [3. The inference loop](#3-the-inference-loop)
  - [4. On-device calibration](#4-on-device-calibration)
  - [5. Persisting prototypes](#5-persisting-prototypes)
  - [STM32 X-CUBE-AI instead of TFLite Micro](#stm32-x-cube-ai-instead-of-tflite-micro)
  - [STM32N6 is a different target, not a faster one](#stm32n6-is-a-different-target-not-a-faster-one)
- [Flashing](#flashing)
- [Verifying the port](#verifying-the-port)
- [How it works / notes](#how-it-works--notes)

---

## Install

The export toolchain is an optional dependency group (not needed for the core library
or realtime paths):

```bash
pip install -e ".[deploy]"      # onnx, onnxruntime, tensorflow, onnx2tf, tf_keras
```

## Quick start

```bash
# recommended: the deployable prototypical model (emits the C runtime too)
python examples/deployment/export_model.py --model EmagerCNNProtoRingStrided

# a plain classifier (no prototypes, no C runtime)
python examples/deployment/export_model.py --model EmagerCNNRingStrided

# re-export the exact weights of a previous run, byte-identically
python examples/deployment/export_model.py --model EmagerCNNProtoRingStrided \
    --checkpoint examples/deployment/exported/EmagerCNNProtoRingStrided/model_fp32.pt

# only some artifacts
python examples/deployment/export_model.py --formats onnx tflite-int8 carray

# reference data to check the port once it is on the board
python examples/deployment/make_test_vectors.py --model EmagerCNNProtoRingStrided
```

Key flags:

| Flag | Meaning |
|---|---|
| `--model` | model class from [`new_emager_cnn.py`](../../emagerlib/models/new_emager_cnn.py) |
| `--dataset` | dataset folder under `Datasets/` (default `Test_EM_C7_R5`) |
| `--checkpoint` | export a saved `state_dict` instead of training fresh |
| `--epochs` | training epochs when there is no `--checkpoint` |
| `--proto-export` | `embedding` (default) or `baked` — see [below](#prototypical-models-why-the-graph-stops-at-the-embedding) |
| `--calib-samples` | representative windows for INT8 calibration (default 300) |
| `--verify-samples` | held-out windows used for verification (default 256) |
| `--out`, `--keep-work` | output dir; retain the `_work/` SavedModel intermediates |

> Training is **not** seeded, so re-running the tool gives different weights. The FP32
> `state_dict` that produced each export is saved as `model_fp32.pt` — use
> `--checkpoint` to reproduce an artifact exactly. Without it, the accuracy in
> `EXPORT_REPORT.md` would describe weights that no longer exist.

## Which model should I export?

- **`EmagerCNNProtoRingStrided`** — **the recommended target.** Smallest architecture
  (44 K params), and the prototype rule means per-user calibration needs **no backprop
  on the device**. Measured 98.8% at 5 calibration reps (LOSO, Felix).
- **`EmagerCNNRingStrided`** — the same architecture as a plain classifier, if you do
  not want per-user calibration at all. No C runtime is emitted.

**Do not export the `…PTQ` / `…QAT` variants.** Those quantize themselves through
`torch.ao` during `fit()`, and eager-mode quantized modules have no ONNX lowering. They
are also redundant: this pipeline does its own INT8 via `TFLiteConverter`, calibrated on
real MAV windows, so exporting the FP32 variant gives you the same artifact. The tool
refuses them with that message.

See [`models.md`](../../emagerlib/models/models.md) for the full variant zoo.

---

## Prototypical models: why the graph stops at the embedding

A prototypical model has no learned classifier. It classifies by **nearest prototype**,
where each prototype is the mean embedding of a few examples of a gesture — and those
examples come from *the user, on the device, after flashing*.

So the prototypes cannot live inside the network. Folding them into the graph as
constants would freeze them at flash time and make calibration impossible.

By default (`--proto-export embedding`) the exported graph is the embedding `f_θ`
**alone**, and the prototype distance ships as C source over a writable array:

```
   MAV[64] ──► emager_quantize_input()
                      │  int8[64]
                      ▼
        ┌───────────────────────────┐
        │  model_int8.tflite        │   the network — frozen at flash time
        │  TFLite Micro / CMSIS-NN  │   ~152 K MAC, 54 KiB
        └───────────────────────────┘
                      │  int8[64]
                      ▼
              emager_dequantize_embedding()
                      │  float[64] embedding
                      ▼
        ┌───────────────────────────┐
        │  emager_classify()        │   prototypes — REWRITABLE at runtime
        │  emager_proto.c           │   7 × 64 = 448 MAC, 1.8 KB
        └───────────────────────────┘
                      │
                      ▼            class index
```

The split costs nothing: the network is ~152 K multiply-accumulates, the distance is
448. And it is the same boundary the PyTorch model already draws — the quantized proto
variants bracket only `embed()` with Quant/DeQuant stubs and compute the distance in
FP32.

`--proto-export baked` folds the prototypes in instead: one self-contained `.tflite`, no
C runtime, **no on-device calibration**. Use it only if the device will never be
calibrated.

> **Why prototypes are computed through the INT8 network, not the float one.** On device
> there is no float network, so calibration has no choice. That turns out to also be the
> better option: prototypes sourced from the INT8 embedding beat FP32-sourced ones at
> every calibration budget tested (+0.3 pp mean). Prototypes should be means of the same
> INT8 vectors they are later compared against.

---

## Output artifacts

Everything lands in `examples/deployment/exported/<model>/` (git-ignored):

| File | What it is | Use it for |
|---|---|---|
| `model_int8.tflite` | full-integer INT8 (int8 weights **and** int8 I/O) | TFLite Micro / CMSIS-NN; X-CUBE-AI |
| `emager_model_data.h` | the above as a C byte array (`alignas(8) g_model[]`) | compile straight into firmware |
| `emager_model_params.h` | dims + INT8 scale/zero-point, **read off the real `.tflite`** | generated — never hand-edit |
| `emager_prototypes.h` | factory prototypes (`EMAGER_FACTORY_PROTOTYPES`) | the shipped default |
| `emager_proto.h` / `.c` | classify + on-device calibration | the part you call |
| `emager_selftest.h` / `.c` | replays test vectors through your firmware | verifying the port |
| `emager_test_vectors.h` | **not written by this tool** — see `make_test_vectors.py` | reference data for the above |
| `model.onnx` | ONNX graph, fixed `[1,64]` input | ONNX Runtime; X-CUBE-AI |
| `model_float32.tflite` | float TFLite | debugging / MPU-class targets |
| `model_fp32.pt` | the FP32 weights this export came from | reproduce with `--checkpoint` |
| `EXPORT_REPORT.md` | sizes, per-stage agreement, INT8 quant params | provenance |

Typical verification table (`EmagerCNNProtoRingStrided`, 128 held-out windows):

| Artifact | Size | Top-1 agreement vs PyTorch | Max logit diff |
|---|---|---|---|
| model.onnx | 178 KiB | 100.0% | 1.4e-06 |
| model_float32.tflite | 180 KiB | 100.0% | 1.9e-06 |
| model_int8.tflite | **54 KiB** | 100.0% | 5.4e-01 |

Agreement is the fraction of held-out windows where the artifact's **argmax** equals
PyTorch's. INT8 shows a large max *logit* diff but identical argmax — expected:
quantization shifts logit values while preserving the decision.

For prototypical models the agreement number covers the **whole device path** —
quantize → INT8 embedding → dequantize → prototype distance → argmax — compared against
the PyTorch model's complete forward. Verifying the embedding tensor alone would say
nothing about whether the device predicts the same class.

---

## On the MCU

### Memory budget

| Item | Size | Where |
|---|---|---|
| `g_model[]` (INT8 network) | **54 KiB** | flash (`const`) |
| `EMAGER_FACTORY_PROTOTYPES` | 1.8 KB | flash (`const`) |
| live `emager_prototypes_t` | 1.8 KB | RAM (writable copy) |
| `emager_calib_t` (calibration only) | ~1.8 KB | RAM, can be stack/scratch |
| TFLite Micro tensor arena | ~16 KB to start | RAM |
| embedding + scores | ~300 B | RAM |

The arena figure is a **starting point, not a measurement** — it depends on your TFLM
build. Get the true number on your target:

```c
printf("arena used: %u\n", (unsigned)interpreter.arena_used_bytes());
```

Size it up until `AllocateTensors()` succeeds, then trim to `arena_used_bytes()` plus a
small margin.

### 1. Drop the files in

Copy these from `exported/<model>/` into your firmware project:

```
emager_model_data.h     emager_model_params.h     emager_prototypes.h
emager_proto.h          emager_proto.c
```

`emager_proto.c` is plain C99 — no allocation, no float64, nothing beyond `<string.h>`
and `<math.h>`. Compile it like any other source file.

### 2. Set up TFLite Micro

These are the ops the exported model actually uses (read off the `.tflite`, not
guessed). Registering exactly these keeps the resolver — and the binary — small:

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" {
#include "emager_proto.h"
}
#include "emager_model_data.h"      // g_model[]
#include "emager_prototypes.h"      // EMAGER_FACTORY_PROTOTYPES

constexpr int kArenaSize = 16 * 1024;            // tune via arena_used_bytes()
alignas(16) static uint8_t g_arena[kArenaSize];

static tflite::MicroInterpreter *g_interp = nullptr;

bool emager_model_init() {
    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) return false;

    // Exactly the ops in the exported graph. CONCATENATION/SLICE/SPLIT come from the
    // ring padding (a torch.cat of edge slices); MUL/ADD are the input BatchNorm, which
    // stays in float on the FP32 side of the QuantStub.
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddConcatenation();
    resolver.AddPad();
    resolver.AddSlice();
    resolver.AddSplit();
    resolver.AddTranspose();
    resolver.AddMul();
    resolver.AddAdd();

    static tflite::MicroInterpreter interp(model, resolver, g_arena, kArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) return false;
    g_interp = &interp;
    return true;
}
```

> If `AllocateTensors()` fails, the arena is too small — raise `kArenaSize`. If `Invoke()`
> returns an error about a missing op, add it to the resolver and bump the `<10>`
> template argument. Re-run the export and check `EXPORT_REPORT.md` if you change models:
> a different architecture can need a different op set.

### 3. The inference loop

```c
// Compute the 64 MAV features exactly as training does: mean(|x|) over a 200-sample
// window of the 4x16 grid. Same window size, same increment, same channel order.
static float mav[EMAGER_N_FEATURES];
static float embed[EMAGER_EMBED_DIM];
static emager_prototypes_t g_protos;   // RAM copy — the writable one

void emager_boot(void) {
    // Start from the factory defaults; overwrite with the user's if you have them.
    memcpy(&g_protos, &EMAGER_FACTORY_PROTOTYPES, sizeof(g_protos));
    if (flash_has_prototypes()) {
        flash_read_prototypes(&g_protos);      // your storage — see below
    }
}

int emager_predict(const float mav_in[EMAGER_N_FEATURES]) {
    TfLiteTensor *in  = g_interp->input(0);
    TfLiteTensor *out = g_interp->output(0);

    emager_quantize_input(mav_in, in->data.int8);           // float -> int8
    if (g_interp->Invoke() != kTfLiteOk) return -1;
    emager_dequantize_embedding(out->data.int8, embed);     // int8 -> float embedding

    float scores[EMAGER_N_CLASSES];
    return emager_classify(&g_protos, embed, scores);       // -1 if uncalibrated
}
```

`emager_classify` returns the class index, or `-1` if no prototype is valid. Pass `NULL`
for `scores` if you only want the class; when given, `scores[c] = -‖e − p_c‖²/D`, which
is exactly the logit PyTorch produces, so a softmax over it matches `predict_proba()`.

### 4. On-device calibration

This is the whole point of the prototypical model. The user performs each gesture a few
times; you average the embeddings. **No gradients, no optimizer, no training data kept.**

```c
static emager_calib_t cal;

void calibration_start(void) {
    emager_calib_reset(&cal);
}

// Call for every recorded window while the user holds gesture `cls`.
void calibration_feed(const float mav_in[EMAGER_N_FEATURES], uint8_t cls) {
    TfLiteTensor *in  = g_interp->input(0);
    TfLiteTensor *out = g_interp->output(0);
    emager_quantize_input(mav_in, in->data.int8);
    if (g_interp->Invoke() != kTfLiteOk) return;
    emager_dequantize_embedding(out->data.int8, embed);
    emager_calib_add(&cal, embed, cls);
}

bool calibration_finish(void) {
    emager_prototypes_t fresh;
    if (emager_calib_finalize(&cal, &fresh) != EMAGER_N_CLASSES) {
        return false;                      // a gesture was skipped — keep the old set
    }
    memcpy(&g_protos, &fresh, sizeof(g_protos));
    flash_write_prototypes(&g_protos);
    return true;
}
```

Notes that matter:

- **Finalize into a scratch struct**, as above. Writing straight into the live
  prototypes means an abandoned calibration leaves the device half-configured.
- `emager_calib_finalize` returns how many classes got at least one sample. A class with
  none is marked **invalid**, not zeroed-and-used — a zeroed prototype is not neutral, it
  sits at the origin of embedding space and would attract samples near it. A user who
  skips a gesture must not get a class that fires at random.
- **How many reps?** Measured on Felix: 1 rep already gives 98.7%, and more barely moves
  it (98.8% at 5, 98.9% at 9). On a subject with poor cross-session transfer the gain from
  calibration is far larger. Asking for ~3 reps per gesture is a reasonable default.
- A "shot" is a **repetition**, not a window — feed every window of the rep.

### 5. Persisting prototypes

`sizeof(emager_prototypes_t)` is `EMAGER_PROTOTYPES_NBYTES` — 1.8 KB for the default
7×64 model. Small enough for a flash sector, EEPROM, or an external FRAM. The struct is
POD (floats + bytes), so a raw `memcpy` to storage is fine.

Two things to get right:

- **Version your blob.** Prototypes are only meaningful for the exact network that
  produced them. Store a hash/version of `emager_model_data.h` alongside them and discard
  the saved set if it does not match, or a firmware update silently pairs new weights with
  stale prototypes — which fails quietly, as wrong predictions rather than an error.
- **Don't trust a partial write.** Checksum the blob; fall back to
  `EMAGER_FACTORY_PROTOTYPES` if it fails.

### STM32 X-CUBE-AI instead of TFLite Micro

X-CUBE-AI generates its own C from `model.onnx` or `model_int8.tflite` — feed it either
and it replaces steps 2–3. The prototype half is unchanged: `emager_proto.c` works the
same, you just get the embedding from the X-CUBE-AI network instead of
`interpreter.output(0)`.

```
CubeMX -> Software Packs -> X-CUBE-AI -> Add network
    Model: exported/EmagerCNNProtoRingStrided/model_int8.tflite
    Compression: none (already INT8)   ->  Analyze  ->  Generate code
```

X-CUBE-AI's *Validate on target* is worth running: it reports on-device latency and
checks the network against reference inputs, which the report in this repo cannot do
from the host.

### STM32N6 is a different target, not a faster one

The block above describes classic Cortex-M STM32s (M4/M7/M33), where X-CUBE-AI emits a
software kernel library and the `ai_*` API. **The STM32N6 does not work that way**, and
assuming it does is the fastest way to lose a day:

- The network is compiled for the **Neural-ART NPU** by the ST Edge AI Core / Neural-ART
  compiler, and the generated code is driven by the **LL_ATON** runtime — a different
  API from `ai_network_run()`.
- The N6 has **no internal flash.** Code and `const` data live in external OSPI flash,
  the application boots via an FSBL, and images must be **signed** before they can be
  programmed. Every "put it in a flash sector" instruction on this page needs rethinking
  there — including prototype persistence, which lands in the same external flash the
  firmware itself occupies.
- Ops the NPU does not implement fall back to the Cortex-M55. For this model that
  matters: the ring padding lowers to `CONCATENATION`/`SLICE`/`SPLIT`/`TRANSPOSE` and
  the input BatchNorm stays in float (`MUL`/`ADD`). Run `stedgeai analyze` and read the
  epoch breakdown before assuming the whole graph is accelerated.

Also worth saying plainly: at ~152 K MAC and 54 KiB this model does not need an NPU.
The N6 is a reasonable board to *have*, but the speedup over an M4F running CMSIS-NN
will not be what justifies it — the acquisition and windowing dominate.

Toolchain setup, VS Code configuration and a firmware skeleton for the NUCLEO-N657X0-Q
live in a separate repository, `emager-stm32n6-firmware` — deliberately not in this
one. Nothing embedded belongs in a Python toolbox, and the firmware depends on this
repo only through the files `export_model.py` copies into it.

---

## Flashing

Nothing here is EMaGer-specific — the artifacts are ordinary C. Whatever your board
already uses will work:

```bash
# STM32, ST-LINK
STM32_Programmer_CLI -c port=SWD -w firmware.elf -rst
st-flash write firmware.bin 0x8000000          # open-source alternative

# PlatformIO (any supported board)
pio run -t upload

# Arduino
arduino-cli compile --fqbn <board> . && arduino-cli upload --fqbn <board> -p <port> .
```

**The model lives in flash as `const`** (`g_model[]`, `EMAGER_FACTORY_PROTOTYPES`), so a
firmware image contains everything needed to boot uncalibrated. Only the ~1.8 KB of
user prototypes needs writable storage.

If you keep prototypes in the same flash bank you are rewriting, **a firmware update
erases them.** Either put them in a dedicated sector your linker script protects, or
accept re-calibration after every update and say so in the UI.

---

## Verifying the port

The export tool verifies the artifacts against PyTorch on the host. It cannot verify
*your firmware*.

The quickest way to close that gap is the on-device self-test. Generate reference data
from the export you flashed, then replay it on the board:

```bash
python examples/deployment/make_test_vectors.py --model EmagerCNNProtoRingStrided
#   -> exported/<model>/emager_test_vectors.h   (32 held-out windows + expected embeddings)
```

```c
#include "emager_selftest.h"

emager_selftest_result_t r;
emager_prototypes_t factory;
memcpy(&factory, &EMAGER_FACTORY_PROTOTYPES, sizeof(factory));

if (!emager_selftest_run(my_forward, &factory, &r)) {
    printf("selftest FAILED: embed %u/%u, class %u/%u, worst err %ld LSB on vector %u\n",
           r.n_embed_ok, r.n_vectors, r.n_class_ok, r.n_vectors,
           (long)r.embed_max_abs_err, r.worst_vector);
}
```

`my_forward` is your runtime's `int8[64] -> int8[D]` call — TFLM's `Invoke()`,
X-CUBE-AI's `ai_network_run()`, or the N6's LL_ATON entry point. Passing it as a
function pointer is what keeps the check independent of which one you chose.

Read the two counters separately, because they fail for unrelated reasons:

- **embeddings match, class differs** → the network is fine; the prototypes are not.
  Stale persisted blob, wrong `valid` flags — or the device has simply been calibrated,
  which changes the class for the same embedding *by design*. The class column of the
  vectors only describes the factory prototypes.
- **embeddings differ** → the network path: MAV feature mismatch (window size,
  increment, channel order), wrong input quantization, or a mis-wired runtime.

The tolerance is 2 INT8 LSBs rather than 0 on purpose — the host runs TFLite's reference
kernels, the device runs CMSIS-NN or an NPU, and they need not agree on the last bit of
rounding.

To go further, or if the self-test itself will not build, pin one thing at a time:

1. **Feed a known MAV vector** through `emager_predict()` and compare against the host.
   Reproduce a host-side reference with:
   ```bash
   python examples/deployment/export_model.py --model EmagerCNNProtoRingStrided \
       --checkpoint exported/EmagerCNNProtoRingStrided/model_fp32.pt --verify-samples 8
   ```
2. **Check the embedding before blaming the prototypes.** If `embed[]` matches the host
   but the class does not, the bug is in the prototypes (stale blob? wrong `valid`
   flags?). If `embed[]` itself is wrong, it is the network path — usually the input
   quantization or the MAV feature not matching training (window size, increment,
   channel order).
3. **`emager_classify` returning −1** means no prototype is valid: the device booted
   without factory defaults and was never calibrated.
4. **Silently wrong predictions after a firmware update** — almost always stale
   prototypes paired with new weights. See the versioning note above.

The generated C is compiled clean under `-std=c99 -Wall -Wextra -Werror` and reproduces
PyTorch's predictions exactly (100% on 128 held-out windows) when fed the real INT8
embeddings.

---

## How it works / notes

- **ONNX** is exported with the legacy TorchScript exporter (`dynamo=False`, opset 17) —
  a plain static graph that `onnx2tf` consumes reliably.
- **TFLite** goes ONNX → TF SavedModel via `onnx2tf`, then TensorFlow's own
  `TFLiteConverter` does the INT8 quantization. onnx2tf's built-in `-oiqt` quantizer was
  tried first, but its flatbuffer-direct path defaults activations to `[-1, 1]` (ignoring
  the representative data), which clips real MAV / logit ranges — so calibration uses
  `TFLiteConverter.representative_dataset` on real MAV windows instead.
- **The input BatchNorm stays in float.** It sits on the FP32 side of the QuantStub in the
  PyTorch model (it normalizes the raw signal range), which is why `MUL`/`ADD` appear in
  the op list.
- **Ring padding becomes `CONCATENATION`/`SLICE`/`SPLIT`.** `RingPad2d` wraps the
  electrode-column axis with a `torch.cat` of edge slices rather than
  `F.pad(mode="circular")` — the cat form is quantization-safe and lowers cleanly.
- **Verification** runs one sample at a time (batch 1) through each artifact and compares
  argmax + logits against the PyTorch reference.
- The generated headers are written as **ASCII**, deliberately: they go into embedded
  toolchains, and encoding failures should surface at generation time rather than on the
  target's compiler.
