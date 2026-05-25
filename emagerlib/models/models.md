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

All variants share the Lightning + LibEMG boilerplate via `_EmagerBase`.
Only `__init__` (the architecture) differs between them.

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

## Summary table

| Model | Key difference | Params | File size | Test acc* |
|---|---|---|---|---|
| Legacy (`models.py`) | Conv→**ReLU→BN** (non-standard) | 562 K | 2.2 MB | 99.8% |
| `emager_cnn.py` (reference) | Conv→BN→ReLU (standard) | 562 K | 2.2 MB | 99.3% |
| `EmagerCNNBase` | Identical to reference | 562 K | 2.2 MB | 99.7% |
| `EmagerCNNWide` | 64-ch convs, bigger FC | 1.17 M | 4.7 MB | 99.6% |
| `EmagerCNNDeep` | Extra 3×3 conv layer | 572 K | 2.3 MB | 99.6% |
| `EmagerCNNLight` | 16-ch convs, 128-dim FC | 141 K | 0.6 MB | 99.3% |
| `EmagerCNNStrided` | Stride=2 collapses spatial | 44 K | 0.18 MB | **99.5%** |
| `EmagerCNNCircular` | Ring padding on W only (column axis) | 562 K | 2.2 MB | TBD |
| `EmagerCNNRingStrided` | Strided + ring padding on W only | 44 K | 0.18 MB | TBD |
| `EmagerCNNGAP` | Global avg pool, no FC | 36 K | 0.14 MB | 98.7% |

*Tested on `Test_EM_C7_R5` (7 classes, 5 reps, train=[0-3], test=[4], 10 epochs, CPU).

> The dominant cost in Base / Wide / Deep / Circular is `Linear(2048, 256)` (~525 K params, ~2.1 MB).
> **EmagerCNNStrided** is the best size/accuracy tradeoff so far: 12× fewer params than Base for only −0.2% accuracy.
> **EmagerCNNCircular** previously underperformed (98.2%) because PyTorch's `padding_mode='circular'` wraps both H and W — but only the column axis is a physical ring. The current implementation uses `RingPad2d` (W-circular, H-zero) for the physically correct prior; pending re-test.
> **EmagerCNNRingStrided** combines both fixes: strided collapse + ring padding on W only.

---

## Adding a new variant

1. Add a class to `new_emager_cnn.py` that inherits from `_EmagerBase`
2. Only define `__init__` — write the full architecture explicitly (no abstraction)
3. Add a row to the summary table and a section here describing what changed and why
