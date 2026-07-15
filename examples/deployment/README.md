# Microcontroller export (`export_model.py`)

Convert a trained EMaGer CNN into microcontroller-ready artifacts and verify,
at every stage, that the converted model still matches the PyTorch original.

```
PyTorch  ->  ONNX  ->  TFLite float32  ->  TFLite full-INT8  ->  C array
             |             |                    |                   |
        model.onnx  model_float32.tflite  model_int8.tflite   model_int8.h
```

Every exported model takes a single **`(1, 64)` float32 MAV feature vector**
(mean-absolute-value over a 200-sample window on the 4×16 electrode grid) and
returns per-class logits. Batch size is fixed to 1 — one inference at a time,
the way an MCU runs it.

## Install

The export toolchain is an optional dependency group (not needed for the core
library or realtime paths):

```bash
pip install -e ".[deploy]"      # onnx, onnxruntime, tensorflow, onnx2tf, tf_keras
```

## Run

```bash
# default: train EmagerCNNRingStrided on Test_EM_C7_R5, export all formats
python examples/deployment/export_model.py

# pick a variant / dataset
python examples/deployment/export_model.py --model EmagerCNNRingStrided --dataset Test_EM_C7_R5

# export a specific already-trained checkpoint instead of training fresh
python examples/deployment/export_model.py --model EmagerCNNBase --checkpoint path/to/state_dict.pt

# only some artifacts
python examples/deployment/export_model.py --formats onnx tflite-int8 carray
```

Key flags: `--epochs` (training when no `--checkpoint`), `--calib-samples`
(representative windows for INT8 calibration, default 300), `--verify-samples`
(held-out windows used for verification, default 256), `--out` (output dir),
`--keep-work` (retain the `_work/` SavedModel intermediates).

## Output

Artifacts land in `examples/deployment/exported/<model>/` (git-ignored):

| File | What it is | Use it for |
|---|---|---|
| `model.onnx` | ONNX graph, fixed `[1,64]` input | ONNX Runtime; STM32 X-CUBE-AI |
| `model_float32.tflite` | float TFLite | debugging / MPU-class targets |
| `model_int8.tflite` | full-integer INT8 (int8 weights **and** int8 I/O) | TFLite Micro / CMSIS-NN; STM32 X-CUBE-AI |
| `model_int8.h` | `model_int8.tflite` as a C byte array (`alignas(8)`) | compile straight into a TFLite Micro firmware |
| `EXPORT_REPORT.md` | sizes, per-stage agreement, INT8 I/O quant params | provenance |

The tool also prints a verification table, e.g. for `EmagerCNNRingStrided`:

| Artifact | Size | Top-1 agreement vs PyTorch | Max logit diff |
|---|---|---|---|
| model.onnx | 189 KiB | 100.0% | 3.7e-06 |
| model_float32.tflite | 182 KiB | 100.0% | 3.8e-06 |
| model_int8.tflite | 55 KiB | 99.6% | 1.7e+00 |

(Agreement is the fraction of held-out windows where the artifact's argmax equals
PyTorch's. INT8 shows a large max *logit* diff but near-perfect argmax — that is
expected: quantization shifts the logit values but preserves the decision.)

## Feeding the INT8 model on the MCU

INT8 TFLite uses affine quantization: `real = scale * (q_int8 − zero_point)`.
The tool reports the exact input/output scale & zero-point (also in
`EXPORT_REPORT.md`). On device:

```c
// input : quantize the 64 float MAV values before invoke()
q_in[i] = (int8_t)roundf(mav[i] / in_scale) + in_zp;     // clamp to [-128, 127]

// output: dequantize the logits after invoke() (only needed if you want scores;
//         argmax over the raw int8 outputs already gives the predicted class)
logit[c] = (out[c] - out_zp) * out_scale;
```

## Which model to export

- **`EmagerCNNRingStrided`** — smallest architecture variant (strided spatial
  collapse removes the dominant FC layer); the recommended first target for an MCU.
- **`EmagerCNNProtoCE` / `EmagerCNNProtoEpisodic`** — few-shot prototypical models.
  The tool bakes in generic "factory" prototypes computed from the training reps so
  the exported distance-based forward is self-contained. (On-device few-shot
  re-calibration — recomputing prototypes from a handful of new windows — is a
  firmware-side step, not part of this export.)

See [`emagerlib/models/models.md`](../../emagerlib/models/models.md) for the full
variant zoo, sizes, and accuracy tables.

## How it works / notes

- **ONNX** is exported with the legacy TorchScript exporter (`dynamo=False`,
  opset 17) — a plain static graph that `onnx2tf` consumes reliably.
- **TFLite** goes ONNX → TF SavedModel via `onnx2tf`, then TensorFlow's own
  `TFLiteConverter` does the INT8 quantization. onnx2tf's built-in `-oiqt`
  quantizer was tried first but its flatbuffer-direct path defaults activations to
  a `[-1, 1]` range (ignoring the representative data), which clips the real MAV /
  logit ranges — so calibration is done with `TFLiteConverter.representative_dataset`
  on real MAV windows instead.
- Verification runs each sample one at a time (batch 1) through the artifact and
  compares argmax + logits against the PyTorch reference.
