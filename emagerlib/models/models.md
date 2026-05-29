# Models

Overview of all model files, their status, architecture, and estimated size.

---

## File map

| File | Status | Purpose |
|---|---|---|
| `models.py` | Legacy | Original implementation — keep as reference, do not use in new code |
| `emager_cnn.py` | Stable | Clean reference model — production baseline |
| `new_emager_cnn.py` | Experimental | Architecture variants for testing and optimization |

---

## `models.py` — Legacy

Contains the original `EmagerCNN` and `EmagerSCNN`. **Do not use for new work.**

Problems:
- Quantization branches (`brevitas`) mixed in with the normal path
- Large blocks of commented-out code from earlier experiments
- `nn.Identity()` input wrapper that does nothing in the non-quantized path

Kept in the repo as a reference for the quantization / FINN deployment path.

### EmagerCNN (original)

```
input → BN1d → reshape(B,1,H,W)
  → Conv(1→32,  3x3) → ReLU → BN2d      ← non-standard order
  → Conv(32→32, 3x3) → ReLU → BN2d      ← non-standard order
  → Conv(32→32, 5x5) → ReLU → BN2d      ← non-standard order
  → Flatten
  → Linear(32·H·W → 256) + Dropout(0.5) + BN1d + ReLU
  → Linear(256 → C)
```

> The conv layers apply ReLU **before** BN (`self.bn1(self.relu1(self.conv1(out)))`).
> The standard order is Conv→BN→ReLU (as in `emager_cnn.py` and all new variants).
> Same parameter count (562 K) — different computation order only.

### EmagerSCNN (original)

Siamese version using `TripletMarginLoss`. Produces embeddings instead of class logits.
`test_step` is commented out — incomplete.

---

## `emager_cnn.py` — Stable reference

Clean rewrite of `EmagerCNN`. No quantization, no dead code.
This is the **baseline** — all experiments in `new_emager_cnn.py` should be compared against this.

Architecture:

```
input → BN1d → reshape(B,1,H,W)
  → Conv(1→32,  3x3) → BN2d → ReLU      ← standard order (fixed vs legacy)
  → Conv(32→32, 3x3) → BN2d → ReLU
  → Conv(32→32, 5x5) → BN2d → ReLU
  → Flatten
  → Linear(32·H·W → 256) + Dropout(0.5) + BN1d + ReLU
  → Linear(256 → C)
```

> Same parameter count as the legacy model (562 K) but conv layers use the correct
> Conv→BN→ReLU order. `EmagerCNNBase` in `new_emager_cnn.py` is architecturally identical to this.

Key details:
- `save_hyperparameters()` — checkpoints are reloadable by hparams
- `_shared_step` — train / val / test share the same logic
- `self.eval()` in `predict_proba` — BatchNorm uses running stats at inference
- Full LibEMG interface: `predict`, `predict_proba`, `fit`

**Estimated size: ~2.2 MB** (float32, input 4×16)

---

## `new_emager_cnn.py` — Experimental variants

All variants share the Lightning + LibEMG boilerplate via `_EmagerBase`. The
**architecture variants** change only `__init__` (the layers). The **quantization
variants** also override the training pipeline (`fit`) to fuse, quantize to INT8
and evaluate on CPU, and form a small subclass chain:
`EmagerCNNQuantizedPTQ` → `EmagerCNNQuantizedQAT` → `EmagerCNNRingStridedQAT`.

The dominant cost in most models is the **FC hidden layer** (`Linear(32·H·W, 256)`).
For the default 4×16 input that's `Linear(2048, 256)` — ~525k of the ~562k total params.

---

### EmagerCNNBase

Identical to `emager_cnn.py`. Use as the control group.

```
Conv(1→32, 3x3) → Conv(32→32, 3x3) → Conv(32→32, 5x5) → Flatten → Linear(2048→256) → Linear(256→C)
```

**~562k params — ~2.2 MB**

---

### EmagerCNNWide

Second and third conv widened to 64 channels. More feature maps, same depth and structure.
FC input grows to `64·H·W = 4096`.

```
Conv(1→32, 3x3) → Conv(32→64, 3x3) → Conv(64→64, 5x5) → Flatten → Linear(4096→256) → Linear(256→C)
                         ↑ 64                  ↑ 64                          ↑ 4096
```

**~1.17M params — ~4.7 MB**

---

### EmagerCNNDeep

Extra 3×3 conv inserted before the final 5×5 (4 layers total vs 3 in Base).
Slightly more params but the same FC cost.

```
Conv(1→32, 3x3) → Conv(32→32, 3x3) → Conv(32→32, 3x3) → Conv(32→32, 5x5) → Flatten → Linear(2048→256) → Linear(256→C)
                                              ↑ extra layer
```

**~572k params — ~2.3 MB**

---

### EmagerCNNLight

Half the channels (16) and half the FC dim (128). Roughly 4× fewer parameters than Base.

```
Conv(1→16, 3x3) → Conv(16→16, 3x3) → Conv(16→16, 5x5) → Flatten → Linear(1024→128) → Linear(128→C)
      ↑ 16               ↑ 16                ↑ 16                          ↑ 1024/128
```

**~141k params — ~0.6 MB**

---

### EmagerCNNStrided

Stride=2 on the first two convolutions collapses the spatial dimensions before flattening.
For a 4×16 input: `(4,16) → (2,8) → (1,4)` — flatten gives 128 instead of 2048.
The FC hidden layer shrinks accordingly, removing ~95% of the total parameter count vs Base.

```
Conv(1→32, 3x3, s=2) → Conv(32→32, 3x3, s=2) → Conv(32→32, 5x5) → Flatten(128) → Linear(128→64) → Linear(64→C)
           ↓ (2,8)              ↓ (1,4)
```

**~44k params — ~0.18 MB**

---

### EmagerCNNCircular

Same architecture as Base — only the padding scheme changes.
Uses `RingPad2d` (W-circular, H-zero) before each conv, then `Conv2d(padding=0)`.
Physically motivated: the 16 electrode columns sit on a ring, so column 15 is adjacent to column 0.

Earlier version used PyTorch's `padding_mode='circular'`, which wraps **both** H and W.
The row axis (4 rows) does not physically loop, so wrapping it injected non-physical
correlations — that variant underperformed Base (98.2%). This version pads the
row axis with zeros (matching Base) and only wraps the electrode column axis.

```
RingPad(w=1,h=1) → Conv(1→32, 3x3, p=0) + BN + ReLU
RingPad(w=1,h=1) → Conv(32→32, 3x3, p=0) + BN + ReLU
RingPad(w=2,h=2) → Conv(32→32, 5x5, p=0) + BN + ReLU
Flatten → Linear(2048→256) → Linear(256→C)
```

**~562k params — ~2.2 MB** (identical to Base — RingPad2d has no params)

---

### EmagerCNNRingStrided

Combines two of the better techniques in the file: strided spatial collapse
(from `EmagerCNNStrided`) and **per-axis** circular padding — circular on the
column axis (W) only, zero on the row axis (H).

This addresses the issue noted under `EmagerCNNCircular`: PyTorch's
`padding_mode='circular'` wraps both H and W, but only the column axis is a
physical ring on the device. `RingPad2d` does W-circular + H-zero manually,
then the Conv2d uses `padding=0`.

Same flatten size and parameter count as `EmagerCNNStrided` — the only
difference is *which* values pad the conv inputs.

```
RingPad(w=1,h=1) → Conv(1→32, 3x3, s=2, p=0) → (2,8)
RingPad(w=1,h=1) → Conv(32→32, 3x3, s=2, p=0) → (1,4)
RingPad(w=2,h=2) → Conv(32→32, 5x5, s=1, p=0) → (1,4)
Flatten(128) → Linear(128→64) → Linear(64→C)
```

**~44k params — ~0.18 MB** (identical to Strided)

---

### EmagerCNNGAP

Global Average Pooling replaces the FC hidden layer entirely.
After the conv stack, each feature map is averaged to a scalar → 32-dim vector → directly to output.
This removes ~525k parameters (the entire `Linear(2048, 256)` block).

```
Conv(1→32, 3x3) → Conv(32→32, 3x3) → Conv(32→32, 5x5) → AvgPool → Flatten(32) → Linear(32→C)
                                                                ↑ no hidden FC
```

Also input-size agnostic — works on any spatial dimension without recomputing FC sizes.

**~36k params — ~0.14 MB**

---

### EmagerCNNQuantizedPTQ

Optimization variant — **architecture is identical to `EmagerCNNBase`**, only the
weight representation changes. Use this row to isolate the effect of post-training
INT8 quantization independently of any architectural change.

Pipeline (runs inside `fit()`):
1. FP32 training to convergence (standard Lightning loop)
2. Fold the classifier's `BatchNorm1d(256)` into the preceding `Linear(2048→256)`
   — algebraically exact in eval mode (BN uses running stats only). Removes a layer
   that has no `QuantizedCPU` kernel in eager-mode PyTorch.
3. Fuse `Conv+BN+ReLU` triplets in the conv stack (`torch.ao.quantization.fuse_modules`)
4. Calibrate observers on ~10 train batches
5. `convert()` to INT8 — Conv2d, Linear, BN2d, ReLU all replaced with quantized variants

```
input → BN1d (FP32, raw signal range)
  → reshape → QuantStub (→ INT8 from here)
  → [Conv(1→32, 3x3) + BN2d + ReLU]   fused → INT8
  → [Conv(32→32, 3x3) + BN2d + ReLU]  fused → INT8
  → [Conv(32→32, 5x5) + BN2d + ReLU]  fused → INT8
  → Flatten
  → Linear_folded(2048→256)           INT8 (BN1d folded in)
  → Dropout (no-op in eval)
  → ReLU                              INT8
  → Linear(256→C)                     INT8
  → DeQuantStub (→ FP32 out)
```

Backend defaults to `fbgemm` (x86 host); set `EmagerCNNQuantizedPTQ.qbackend = "qnnpack"`
for ARM (e.g. Cortex-M) deployment.

Notes:
- The classifier `BatchNorm1d` is folded, not retained, so the post-quantization
  module count differs from Base — `Linear → Dropout → BN1d → ReLU → Linear`
  becomes `Linear_folded → Dropout → ReLU → Linear`. This is an exact equivalent
  at inference time.
- Theoretical MACs are unchanged vs. Base (same architecture). The win is in
  weight precision (INT8 ≈ 4× smaller than FP32) and kernel speed (fbgemm /
  qnnpack INT8 paths). Measure latency, not FLOPs, to see the speedup.
- After `convert()`, quantized weights are stored as packed `_packed_params`
  rather than `nn.Parameter`, so counting params on the post-`fit()` model
  under-reports — it misses the dominant `Linear(2048→256)` weight and makes
  this variant look *smaller* than Base despite an identical architecture. The
  benchmark therefore counts params from `model.parameters()` on the **FP32
  model before `fit()`** (same pattern as MACs), so the reported count reflects
  the true architecture and matches `EmagerCNNBase`.

**~562k params — ~0.55 MB INT8** (params reflect the FP32 architecture, identical
to Base; ~4× smaller on disk than Base's 2.2 MB as weights drop FP32 → INT8 — see
the **size caveat** under the summary table)

---

### EmagerCNNQuantizedQAT

INT8 **quantization-aware training (QAT)** — same `EmagerCNNBase` architecture as
`EmagerCNNQuantizedPTQ`, but the quantization error is simulated *during* training
with fake-quant observers so the weights learn to be robust to it. Subclasses
`EmagerCNNQuantizedPTQ` and reuses its scaffold; only the training pipeline differs.

Pipeline (inside `fit()`):
1. FP32 warm start — converges and populates the BN running stats the fold relies on
2. Fold classifier `BatchNorm1d` into the preceding `Linear` (exact in eval mode)
3. Fuse `Conv+BN+ReLU` with `fuse_modules_qat` → `ConvBnReLU2d` (BN stays trainable)
4. `prepare_qat()` inserts fake-quant + observers
5. Fine-tune `qat_epochs` (default 5) on CPU with fake quant active
6. `convert()` → real INT8, then evaluate on CPU

PTQ quantizes *after* training; QAT fine-tunes *with* fake quant. QAT usually
recovers accuracy that PTQ leaves on the table — compare the two rows to measure it.

**~562k params — ~0.55 MB INT8** (identical architecture and size to PTQ; total
wall time is higher — it runs an FP32 warm start *plus* `qat_epochs` of fine-tuning)

---

### EmagerCNNRingStridedQAT

All three optimization axes stacked: **strided** spatial collapse + **column-only
ring padding** + **INT8 QAT**. It is the `EmagerCNNRingStrided` architecture trained
with the `EmagerCNNQuantizedQAT` pipeline. Subclasses `EmagerCNNQuantizedQAT`;
`__init__` swaps in the RingStrided conv stack, and `_fuse_groups` is overridden
because `RingPad2d` sits between the conv triplets (Conv+BN+ReLU at indices 1-3,
5-7, 9-11 instead of 0-2, 3-5, 6-8).

```
input → BN1d → reshape → QuantStub
  RingPad(w=1,h=1) → [Conv(1→32, 3x3, s=2) + BN + ReLU]  fused → INT8 → (2,8)
  RingPad(w=1,h=1) → [Conv(32→32, 3x3, s=2) + BN + ReLU] fused → INT8 → (1,4)
  RingPad(w=2,h=2) → [Conv(32→32, 5x5, s=1) + BN + ReLU] fused → INT8 → (1,4)
  Flatten(128) → Linear_folded(128→64) → ReLU → Linear(64→C) → DeQuantStub
```

Required a quantization-safe `RingPad2d`: its W-circular wrap now uses `torch.cat`
of edge slices instead of `F.pad(mode="circular")`, which raises on quantized int8
tensors. The two are identical for FP32, so `EmagerCNNCircular` /
`EmagerCNNRingStrided` are unaffected.

**~44k params — ~0.057 MB INT8** (RingStrided's ~12× param cut combined with INT8 —
the smallest deployable variant in the file)

---

## Summary table

| Model | Key difference | Params | File size | Overall acc (LOO)* |
|---|---|---|---|---|
| Legacy (`models.py`) | Conv→**ReLU→BN** (non-standard) | 562 K | 2.2 MB | not benchmarked |
| `emager_cnn.py` (reference) | Conv→BN→ReLU (standard) | 562 K | 2.2 MB | not benchmarked |
| `EmagerCNNDeep` | Extra 3×3 conv layer | 572 K | 2.3 MB | **94.5%** |
| `EmagerCNNBase` | Identical to reference | 562 K | 2.2 MB | 94.1% |
| `EmagerCNNLight` | 16-ch convs, 128-dim FC | 141 K | 0.6 MB | 94.0% |
| `EmagerCNNCircular` | Ring padding on W only (column axis)† | 562 K | 2.2 MB | 93.7%† |
| `EmagerCNNWide` | 64-ch convs, bigger FC | 1.17 M | 4.7 MB | 93.6% |
| `EmagerCNNStrided` | Stride=2 collapses spatial | 44 K | 0.18 MB | 93.1% |
| `EmagerCNNRingStrided` | Strided + ring padding on W only | 44 K | 0.18 MB | TBD |
| `EmagerCNNQuantizedPTQ` | INT8 PTQ on Base architecture | 562 K | 0.55 MB§ | 98.2%‡ |
| `EmagerCNNQuantizedQAT` | INT8 QAT on Base architecture | 562 K | 0.55 MB§ | TBD |
| `EmagerCNNRingStridedQAT` | RingStrided arch + INT8 QAT | 44 K | 0.057 MB§ | TBD |
| `EmagerCNNGAP` | Global avg pool, no FC | 36 K | 0.14 MB | 90.4% |

*Leave-one-out cross-validation across reps, averaged over seeds `[42, 123, 456]` on 3 datasets
(`Test_EM_C7_R5`, `Test_EM_C7_R5_02`, `Test_EM_C7_R5_03`), 7 classes × 5 reps each, 10 epochs.
315 total fits, ~3h 55m wall time at ~45s/fit. See breakdown below.

†`EmagerCNNCircular` was re-implemented post-benchmark to use `RingPad2d` (W-circular,
H-zero) instead of PyTorch's both-axes `padding_mode='circular'`. The 93.7% number is
from the prior both-axes version and needs to be re-measured.

‡`EmagerCNNQuantizedPTQ` was added after the main benchmark run. The 98.2% number is from a
single-dataset (`Test_EM_C7_R5`) single-seed (`42`) run only — not directly comparable to
the multi-dataset overall column for the other models. On the same single-dataset/seed run,
`EmagerCNNBase` scored 98.3%, so the quantization-induced accuracy loss is ~0.1 pp. Pending
a full multi-dataset / multi-seed re-run for a fair overall number. The two QAT variants
(`EmagerCNNQuantizedQAT`, `EmagerCNNRingStridedQAT`) have only been smoke-tested so far —
accuracy is TBD pending a real run.

§ **Quantized size caveat.** File sizes for the INT8 variants are the *real serialized
`state_dict` size* (`torch.save`), ≈ ¼ of the FP32 equivalent. An earlier version of the
benchmark summed `numel × element_size` over `state_dict` tensors, which silently dropped
the packed INT8 weights (stored as `_packed_params`, not plain tensors) and reported a
near-constant ~35 KB for *every* quantized model regardless of architecture. That bug is
fixed — the harness now measures the serialized size — so 0.55 MB (Base arch) and 0.057 MB
(RingStrided arch) are the true on-disk footprints, not the old ~35 KB artifact.

### Per-dataset breakdown (LOO, mean ± std over reps × seeds)

| Model | Test_EM_C7_R5 | Test_EM_C7_R5_02 | Test_EM_C7_R5_03 | Overall |
|---|---|---|---|---|
| EmagerCNNDeep     | 98.5% ± 1.5% | 93.4% ± 3.2% | 91.5% ± 6.2%  | **94.5%** |
| EmagerCNNBase     | 98.7% ± 1.5% | 93.0% ± 4.3% | 90.6% ± 8.4%  | 94.1% |
| EmagerCNNLight    | 98.3% ± 2.8% | 93.3% ± 4.0% | 90.4% ± 7.2%  | 94.0% |
| EmagerCNNCircular†| 98.8% ± 1.5% | 92.0% ± 4.5% | 90.4% ± 7.6%  | 93.7%† |
| EmagerCNNWide     | 98.7% ± 1.5% | 92.3% ± 3.4% | 89.8% ± 8.7%  | 93.6% |
| EmagerCNNStrided  | 98.4% ± 1.6% | 90.4% ± 5.5% | 90.4% ± 7.2%  | 93.1% |
| EmagerCNNRingStrided | TBD | TBD | TBD | TBD |
| EmagerCNNQuantizedPTQ‡ | 98.2% ± 1.5% | TBD | TBD | TBD |
| EmagerCNNQuantizedQAT‡ | TBD | TBD | TBD | TBD |
| EmagerCNNRingStridedQAT‡ | TBD | TBD | TBD | TBD |
| EmagerCNNGAP      | 93.9% ± 10.2% | 90.7% ± 6.0% | 86.6% ± 9.2% | 90.4% |

### Takeaways

> The dominant cost in Base / Wide / Deep / Circular is `Linear(2048, 256)` (~525 K params, ~2.1 MB).
>
> **EmagerCNNDeep** edges out Base by +0.4% for only ~10K extra params — one extra 3×3 conv is a cheap win.
> **EmagerCNNWide** does not pay off: 2× the params of Base for −0.5% accuracy.
> **EmagerCNNLight** is the best small-model accuracy: ~4× fewer params than Base for only −0.1% accuracy. Beats Strided by +0.9%.
> **EmagerCNNStrided** remains the best size/accuracy ratio (12× fewer params than Base for −1.0% accuracy), but the LOO penalty is larger than the single-split number suggested.
> **EmagerCNNCircular** held up under LOO (−0.4% vs Base) on the prior both-axes implementation. Since then the padding was rewritten to wrap W only (the physical electrode ring); re-benchmark pending. **EmagerCNNRingStrided** stacks the same W-only ring padding on top of `EmagerCNNStrided` and is also pending its first benchmark.
> **EmagerCNNGAP** is the only clear loser: −3.7% vs Base, and a large std on `Test_EM_C7_R5` (±10.2%) — removing the FC hidden layer costs too much capacity.
> Variance grows on `_02` and `_03` (std up to ±9% on some models) — these splits are noticeably harder than the original `Test_EM_C7_R5`.
> **Quantization (PTQ / QAT).** `EmagerCNNQuantizedPTQ` (preliminary, 1 dataset / 1 seed) drops ~0.1 pp vs Base on the same split while shrinking the model ~4× (2.2 MB → ~0.55 MB INT8). These are the variants that target *weight precision* rather than architecture, and the win composes with the architectural ones: `EmagerCNNRingStridedQAT` stacks INT8 QAT on top of the RingStrided collapse for the smallest deployable model (~44 K params, ~0.057 MB). `EmagerCNNQuantizedQAT` exists to measure how much accuracy QAT recovers over PTQ at the same size. All three INT8 variants are pending a full multi-dataset / multi-seed run.

### Benchmark harness columns

`train_cnn_benchmark.py` now reports four post-training columns per model:

- **Params** — total `model.parameters()` count, taken on the **FP32 model before `fit()`**. For `EmagerCNNQuantizedPTQ` this matters: after PTQ the weights move into packed `_packed_params` (not `nn.Parameter`), so a post-`fit()` count would under-report and wrongly show fewer params than Base for an identical architecture. Counting pre-`fit()` reports the true architectural size.
- **Size (KB)** — real serialized size of the `state_dict` via `torch.save`. Drops ~4× for INT8 vs FP32 (weights go FP32 → INT8). Previously this summed `numel × element_size` over `state_dict` tensors, which dropped the packed INT8 weights and reported a bogus near-constant ~35 KB for every quantized model — see the **§ size caveat** above.
- **MACs** — multiply-accumulates per single forward pass, counted on the **FP32 model before `fit()`** (Conv2d + Linear only). Identical for `EmagerCNNQuantizedPTQ` and `EmagerCNNBase` by construction — quantization does not change op count, only precision.
- **Lat (ms)** — median single-sample inference time via `torch.utils.benchmark.Timer`, 1 thread, batch=1, host-CPU x86 (fbgemm for INT8). Reflects the kernel-speed side of the quantization win that MACs cannot see. Absolute values do not translate to ARM deployment — relative ordering does.

---

## Adding a new variant

1. Add a class to `new_emager_cnn.py` that inherits from `_EmagerBase`
2. Only define `__init__` — write the full architecture explicitly (no abstraction)
3. Add a row to the summary table and a section here describing what changed and why

For an **INT8 variant** of a new architecture, subclass `EmagerCNNQuantizedPTQ`
(PTQ) or `EmagerCNNQuantizedQAT` (QAT) instead: build the conv stack in `__init__`
(ReLU `inplace=False`, plus the `QuantStub`/`DeQuantStub`) and override the
`_fuse_groups` class attribute to point at the Conv+BN+ReLU index triplets in your
`features` layout. `EmagerCNNRingStridedQAT` is the worked example. If the stack
uses ring padding, rely on `RingPad2d` (its `torch.cat` wrap is quantization-safe).
