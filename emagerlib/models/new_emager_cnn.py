import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


# ==============================================================================
#  Padding primitives
# ==============================================================================

class RingPad2d(nn.Module):
    """
    Per-axis padding for the electrode grid:
      - circular on W (electrode columns form a physical ring: col 0 is adjacent to col 15)
      - zero on H (rows do not loop)

    Used by EmagerCNNCircular, EmagerCNNRingStrided and EmagerCNNRingStridedQAT.
    Conv2d after this layer should use padding=0.

    The W-circular wrap is done with torch.cat of edge slices rather than
    F.pad(mode="circular"). The two are identical for FP32 (pad width < W), but
    cat also works on quantized int8 tensors, whereas F.pad's circular mode
    raises on quantized tensors -- which is what lets the QAT variant run.
    """
    def __init__(self, pad_w: int, pad_h: int):
        super().__init__()
        self.pad_w = pad_w
        self.pad_h = pad_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w > 0:
            p = self.pad_w
            x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)  # circular on W (quant-safe)
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0.0)  # zero on H
        return x


# ==============================================================================
#  Quantization primitives
# ==============================================================================

def _fold_bn1d_into_linear(lin: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    Return a single Linear equivalent to `lin` followed by `bn` in eval mode.

    Exact, not an approximation: in eval mode BN1d is the affine map
    y = (x - mean)/sqrt(var + eps) * gamma + beta using running stats only, which
    absorbs into the preceding Linear's weight and bias.

    Needed because PyTorch's QuantizedCPU backend has no nn.BatchNorm1d kernel, so
    a BN1d that receives a quantized tensor has to be folded away before convert().
    (A BN1d on the FP32 side of a QuantStub -- e.g. `normalize` -- is unaffected and
    stays put.)
    """
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    shift = bn.bias - bn.running_mean * scale

    folded = nn.Linear(lin.in_features, lin.out_features, bias=True)
    bias0  = lin.bias.data if lin.bias is not None else torch.zeros(lin.out_features)
    folded.weight.data = lin.weight.data * scale.unsqueeze(1)
    folded.bias.data   = bias0 * scale + shift
    return folded


# ==============================================================================
#  Shared Lightning + LibEMG boilerplate
#  All variants inherit from this — architecture goes in __init__ of each subclass
# ==============================================================================

class _EmagerBase(L.LightningModule):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x.view(x.size(0), -1))
        x = x.view(x.size(0), 1, *self.hparams.input_shape)
        return self.classifier(self.features(x))

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc",  acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    # LibEMG interface
    def convert_input(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x.float().to(self.device)

    def predict_proba(self, x) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return F.softmax(self(self.convert_input(x)), dim=1).cpu().numpy()
        finally:
            self.train(was_training)

    def predict(self, x) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def fit(self, train_dataloader, test_dataloader=None, max_epochs: int = 10):
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        trainer.fit(self, train_dataloader)
        if test_dataloader is not None:
            return trainer.test(self, test_dataloader)


# ==============================================================================
#  Variants — edit / add here
# ==============================================================================

class EmagerCNNBase(_EmagerBase):
    """
    Baseline — matches the reference model (emager_cnn.py).

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ Conv2d(1  -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 5x5) + BN2d + ReLU
      └─ Flatten
      └─ Linear(32·H·W -> 256) + Dropout(0.5) + BN1d + ReLU
      └─ Linear(256 -> C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        n = int(np.prod(input_shape))

        self.normalize  = nn.BatchNorm1d(n)
        self.features   = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * n, 256), nn.Dropout(0.5), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNWide(_EmagerBase):
    """
    Wider second and third conv (32 -> 64). More feature maps, same depth.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ Conv2d(1  -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 64, 3x3) + BN2d + ReLU      <-- 64 instead of 32
      └─ Conv2d(64 -> 64, 5x5) + BN2d + ReLU      <-- 64 instead of 32
      └─ Flatten
      └─ Linear(64·H·W -> 256) + Dropout(0.5) + BN1d + ReLU
      └─ Linear(256 -> C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        n = int(np.prod(input_shape))

        self.normalize  = nn.BatchNorm1d(n)
        self.features   = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * n, 256), nn.Dropout(0.5), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNDeep(_EmagerBase):
    """
    One extra 3x3 conv before the final 5x5 (4 layers total). More spatial abstraction.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ Conv2d(1  -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 3x3) + BN2d + ReLU      <-- extra layer vs Base
      └─ Conv2d(32 -> 32, 5x5) + BN2d + ReLU
      └─ Flatten
      └─ Linear(32·H·W -> 256) + Dropout(0.5) + BN1d + ReLU
      └─ Linear(256 -> C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        n = int(np.prod(input_shape))

        self.normalize  = nn.BatchNorm1d(n)
        self.features   = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * n, 256), nn.Dropout(0.5), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNLight(_EmagerBase):
    """
    Lighter model — half the channels and FC dim. Faster, fewer parameters.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ Conv2d(1  -> 16, 3x3) + BN2d + ReLU      <-- 16 instead of 32
      └─ Conv2d(16 -> 16, 3x3) + BN2d + ReLU      <-- 16 instead of 32
      └─ Conv2d(16 -> 16, 5x5) + BN2d + ReLU      <-- 16 instead of 32
      └─ Flatten
      └─ Linear(16·H·W -> 128) + Dropout(0.5) + BN1d + ReLU    <-- 128 instead of 256
      └─ Linear(128 -> C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        n = int(np.prod(input_shape))

        self.normalize  = nn.BatchNorm1d(n)
        self.features   = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * n, 128), nn.Dropout(0.5), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNGAP(_EmagerBase):
    """
    Global Average Pooling replaces the large FC hidden layer entirely.
    After the conv stack, each of the 32 feature maps is averaged to a single value,
    giving a 32-dim vector fed directly to the output layer.
    Removes ~525k parameters (the dominant cost in Base) with no spatial flattening overhead.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ Conv2d(1  -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 5x5) + BN2d + ReLU
      └─ AdaptiveAvgPool2d(1)  →  (B, 32, 1, 1)
      └─ Flatten               →  (B, 32)
      └─ Linear(32 -> C)       ← no hidden FC layer at all
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.normalize  = nn.BatchNorm1d(int(np.prod(input_shape)))
        self.features   = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, num_classes)
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNStrided(_EmagerBase):
    """
    Strided convolutions replace padding — spatial size shrinks instead of staying constant.
    Flatten output drops from 32·H·W to 32·(H/4)·(W/4), making the FC layer ~16x smaller.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, 4, 16)
      └─ Conv2d(1  -> 32, 3x3, stride=2) + BN2d + ReLU  →  (B, 32, 2, 8)
      └─ Conv2d(32 -> 32, 3x3, stride=2) + BN2d + ReLU  →  (B, 32, 1, 4)
      └─ Conv2d(32 -> 32, 5x5, stride=1) + BN2d + ReLU  →  (B, 32, 1, 4)
      └─ Flatten  →  (B, 128)
      └─ Linear(128 -> 64) + Dropout(0.5) + BN1d + ReLU
      └─ Linear(64 -> C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # compute spatial size after two stride-2 convolutions (kernel=3, padding=1)
        h = (input_shape[0] + 2 - 3) // 2 + 1
        h = (h           + 2 - 3) // 2 + 1
        w = (input_shape[1] + 2 - 3) // 2 + 1
        w = (w           + 2 - 3) // 2 + 1
        strided_flat = 32 * h * w   # 128 for the default (4, 16) input

        self.normalize  = nn.BatchNorm1d(int(np.prod(input_shape)))
        self.features   = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(strided_flat, 64), nn.Dropout(0.5), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNCircular(_EmagerBase):
    """
    Ring padding on all conv layers — treats the electrode column axis (W) as periodic.
    The 16 electrode columns sit on a physical ring, so column 0 is adjacent to column 15.
    The row axis (H) does NOT loop, so it is zero-padded instead of wrapped.

    Uses RingPad2d (W-circular, H-zero) followed by Conv2d(padding=0). PyTorch's built-in
    padding_mode='circular' wraps both H and W, which injects non-physical correlations
    on the H axis — that is why the earlier both-axes variant underperformed Base.

    Architecture is identical to Base — only the padding scheme changes.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ RingPad(w=1, h=1) → Conv2d(1  -> 32, 3x3, padding=0) + BN2d + ReLU
      └─ RingPad(w=1, h=1) → Conv2d(32 -> 32, 3x3, padding=0) + BN2d + ReLU
      └─ RingPad(w=2, h=2) → Conv2d(32 -> 32, 5x5, padding=0) + BN2d + ReLU
      └─ Flatten
      └─ Linear(32·H·W -> 256) + Dropout(0.5) + BN1d + ReLU
      └─ Linear(256 -> C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        n = int(np.prod(input_shape))

        self.normalize  = nn.BatchNorm1d(n)
        self.features   = nn.Sequential(
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(1,  32, kernel_size=3, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            RingPad2d(pad_w=2, pad_h=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * n, 256), nn.Dropout(0.5), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNRingStrided(_EmagerBase):
    """
    Combines two techniques: strided spatial collapse (EmagerCNNStrided) and
    column-only circular padding (a corrected EmagerCNNCircular).

    Why ring padding on W only: the 16 electrode columns sit on a physical ring,
    so column 0 is adjacent to column 15. The H axis (rows) does not loop, so
    wrapping it (as plain padding_mode='circular' does) injects non-physical
    correlations — which is why EmagerCNNCircular underperforms Base.

    Why strided: collapses (4,16) → (1,4) before flatten, removing the dominant
    Linear(2048,256) cost while keeping ~same accuracy as Base.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, 4, 16)
      └─ RingPad(w=1, h=1) → (6, 18)
      └─ Conv2d(1  -> 32, 3x3, stride=2, padding=0) + BN2d + ReLU  →  (B, 32, 2, 8)
      └─ RingPad(w=1, h=1) → (4, 10)
      └─ Conv2d(32 -> 32, 3x3, stride=2, padding=0) + BN2d + ReLU  →  (B, 32, 1, 4)
      └─ RingPad(w=2, h=2) → (5, 8)
      └─ Conv2d(32 -> 32, 5x5, stride=1, padding=0) + BN2d + ReLU  →  (B, 32, 1, 4)
      └─ Flatten  →  (B, 128)
      └─ Linear(128 -> 64) + Dropout(0.5) + BN1d + ReLU
      └─ Linear(64 -> C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # spatial size after two stride-2 convs (kernel=3, pad 1 each side via RingPad)
        h = (input_shape[0] + 2 - 3) // 2 + 1
        h = (h              + 2 - 3) // 2 + 1
        w = (input_shape[1] + 2 - 3) // 2 + 1
        w = (w              + 2 - 3) // 2 + 1
        flat = 32 * h * w   # 128 for the default (4, 16) input

        self.normalize  = nn.BatchNorm1d(int(np.prod(input_shape)))
        self.features   = nn.Sequential(
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            RingPad2d(pad_w=2, pad_h=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(flat, 64), nn.Dropout(0.5), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()


class EmagerCNNQuantizedPTQ(_EmagerBase):
    """
    Identical architecture to EmagerCNNBase. After FP32 training completes,
    applies post-training INT8 quantization (PTQ) via torch.ao.quantization:
    Conv+BN+ReLU triplets are fused, then weights and activations are quantized
    to INT8 using a calibration pass over the training data.

    Use this variant to isolate the impact of *optimization* (weight precision)
    as a separate axis from *architecture*. Compare against EmagerCNNBase to see
    the size/accuracy tradeoff in a like-for-like setting.

    Notes:
      - Backend: fbgemm (x86 dev machine). For ARM deployment (e.g. Cortex-M)
        swap to qnnpack by setting cls.qbackend = "qnnpack" before fit().
      - state_dict size drops ~4x after quantization (FP32 -> INT8 weights).
        Param count via model.parameters() under-reports because quantized
        layer weights are stored as packed _packed_params, not nn.Parameter.
      - The model is moved to CPU before quantization and stays there;
        accuracy is reported from a manual CPU eval loop, not Lightning.test.

    input (B, H*W)
      |- BN1d                                    (kept FP32 -- raw signal range)
      |- reshape (B, 1, H, W)
      |- QuantStub                               (FP32 -> INT8 from here)
      |- [Conv2d(1 -> 32, 3x3) + BN2d + ReLU]    fused
      |- [Conv2d(32 -> 32, 3x3) + BN2d + ReLU]   fused
      |- [Conv2d(32 -> 32, 5x5) + BN2d + ReLU]   fused
      |- Flatten
      |- Linear(32*H*W -> 256)
      |- Dropout(0.5)
      |- [BN1d + ReLU]                           fused
      |- Linear(256 -> C)
      |- DeQuantStub                             (INT8 -> FP32 out)
    """
    qbackend = "fbgemm"  # "qnnpack" for ARM (e.g. STM32 / Cortex-M) deployment
    _calib_batches = 10
    # Conv+BN+ReLU index triplets to fuse in self.features. Matches the plain
    # conv stack (Conv,BN,ReLU at 0-2, 3-5, 6-8). Subclasses with a different
    # features layout (e.g. interleaved RingPad2d) override this.
    _fuse_groups = [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]

    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        n = int(np.prod(input_shape))

        self.normalize  = nn.BatchNorm1d(n)
        self.features   = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * n, 256), nn.Dropout(0.5), nn.BatchNorm1d(256), nn.ReLU(inplace=False),
            nn.Linear(256, num_classes),
        )
        self.criterion  = nn.CrossEntropyLoss()

        self.quant      = torch.ao.quantization.QuantStub()
        self.dequant    = torch.ao.quantization.DeQuantStub()
        self._quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x.view(x.size(0), -1))
        x = x.view(x.size(0), 1, *self.hparams.input_shape)
        x = self.quant(x)
        x = self.classifier(self.features(x))
        x = self.dequant(x)
        return x

    def fit(self, train_dataloader, test_dataloader=None, max_epochs: int = 10):
        # Stage 1: FP32 training (same trainer setup as base)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        trainer.fit(self, train_dataloader)

        # Stage 2: post-training INT8 quantization on CPU
        self._apply_ptq(train_dataloader)

        # Stage 3: evaluate the quantized model on CPU
        if test_dataloader is not None:
            return self._test_quantized(test_dataloader)

    def _apply_ptq(self, calib_loader):
        import torch.ao.quantization as taq

        self.eval()
        self.cpu()

        # Fold the classifier's BatchNorm1d into its preceding Linear, then
        # drop the BN1d module from the Sequential. PyTorch's QuantizedCPU backend
        # has no kernel for nn.BatchNorm1d, so leaving it in place fails after
        # convert(). Folding is exact under eval mode (BN uses running stats only).
        self._fold_classifier_bn()

        # Fuse Conv+BN+ReLU triplets in the conv stack.
        taq.fuse_modules(self.features, self._fuse_groups, inplace=True)

        self.qconfig = taq.get_default_qconfig(self.qbackend)
        taq.prepare(self, inplace=True)

        # Calibration: feed a handful of train batches through observers
        with torch.no_grad():
            for i, (x, _) in enumerate(calib_loader):
                self(x.cpu())
                if i + 1 >= self._calib_batches:
                    break

        taq.convert(self, inplace=True)
        self._quantized = True

    def _fold_classifier_bn(self):
        """
        Fold the classifier's BatchNorm1d into the preceding Linear, then rebuild
        the Sequential without the BN1d module. Valid in eval mode: BN uses running
        stats only, which absorb exactly into the linear's weight and bias.

            Before:  [Linear, Dropout, BN1d, ReLU, Linear]
            After:   [Linear_folded, Dropout, ReLU, Linear]

        Dropout stays in place because it is a no-op in eval mode and passes
        quantized tensors through unchanged.
        """
        lin0    = self.classifier[0]   # Linear(32*n, 256)
        dropout = self.classifier[1]
        bn1d    = self.classifier[2]   # BatchNorm1d(256)
        relu    = self.classifier[3]
        lin1    = self.classifier[4]   # Linear(256, num_classes)

        self.classifier = nn.Sequential(_fold_bn1d_into_linear(lin0, bn1d), dropout, relu, lin1)

    def _test_quantized(self, test_dl):
        self.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in test_dl:
                logits = self(x.cpu())
                loss   = self.criterion(logits, y.cpu())
                correct  += (logits.argmax(dim=1) == y.cpu()).sum().item()
                total    += y.size(0)
                loss_sum += loss.item() * y.size(0)
        return [{"test_acc": correct / total, "test_loss": loss_sum / total}]

    def convert_input(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x.float().cpu() if self._quantized else super().convert_input(x)


class EmagerCNNQuantizedQAT(EmagerCNNQuantizedPTQ):
    """
    Quantization-Aware Training (QAT) counterpart to EmagerCNNQuantizedPTQ (PTQ).

    Same architecture as EmagerCNNBase, but the INT8 quantization error is
    *simulated during training* via fake-quant observers, so the weights learn
    to be robust to it. This usually recovers accuracy that post-training
    quantization (PTQ) leaves on the table — the gap is largest on
    quantization-sensitive models / low bit-widths.

    Inherits architecture, forward, the BN-fold helper, the CPU eval loop and
    convert_input from EmagerCNNQuantizedPTQ. Only the training pipeline differs:
    PTQ quantizes *after* training, QAT fine-tunes *with* fake quant.

    Pipeline (inside fit()):
      1. FP32 training to convergence — a warm start, and it populates the BN
         running stats the classifier fold relies on.
      2. Fold classifier BatchNorm1d into the preceding Linear (exact in eval
         mode; no QuantizedCPU kernel exists for standalone BN1d).
      3. Fuse Conv+BN+ReLU triplets with fuse_modules_qat — produces ConvBnReLU2d,
         which keeps BN trainable through fine-tuning and folds it at convert().
      4. prepare_qat() inserts fake-quant + observers.
      5. Fine-tune for `qat_epochs` epochs on CPU with fake quant active.
      6. convert() to a real INT8 model, then evaluate on CPU.

    Compare against EmagerCNNQuantizedPTQ for the QAT-vs-PTQ accuracy delta, and
    against EmagerCNNBase for the FP32 reference. Backend / size / latency notes
    are identical to EmagerCNNQuantizedPTQ.
    """
    qat_epochs = 5  # fine-tuning epochs after the FP32 warm start

    def fit(self, train_dataloader, test_dataloader=None, max_epochs: int = 10):
        import torch.ao.quantization as taq

        # Stage 1: FP32 warm start (same trainer setup as base)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        trainer.fit(self, train_dataloader)

        # Stage 2-4: fold classifier BN, fuse for QAT, insert fake-quant (on CPU)
        self._prepare_qat()

        # Stage 5: QAT fine-tuning with fake quant active. Forced onto CPU so the
        # fbgemm/qnnpack fake-quant + convert path is consistent on any host.
        qat_trainer = L.Trainer(
            max_epochs=self.qat_epochs,
            accelerator="cpu",
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        qat_trainer.fit(self, train_dataloader)

        # Stage 6: convert fake-quant modules to real INT8, then evaluate on CPU
        self.eval()
        taq.convert(self, inplace=True)
        self._quantized = True
        if test_dataloader is not None:
            return self._test_quantized(test_dataloader)

    def _prepare_qat(self):
        import torch.ao.quantization as taq

        self.eval()
        self.cpu()
        # Fold classifier BN1d now that the FP32 warm start has given it
        # meaningful running stats (exact in eval mode). Reused from the PTQ class.
        self._fold_classifier_bn()

        # QAT-specific fusion: fuse_modules_qat keeps BN inside ConvBnReLU2d so it
        # stays trainable during fine-tuning (plain fuse_modules would fold it away
        # immediately, which is correct for PTQ but defeats QAT).
        self.train()
        taq.fuse_modules_qat(self.features, self._fuse_groups, inplace=True)
        self.qconfig = taq.get_default_qat_qconfig(self.qbackend)
        taq.prepare_qat(self, inplace=True)


class EmagerCNNRingStridedPTQ(EmagerCNNQuantizedPTQ):
    """
    EmagerCNNRingStrided architecture with post-training INT8 quantization (PTQ).

    The PTQ counterpart to EmagerCNNRingStridedQAT: same RingStrided conv stack
    (strided spatial collapse + column-only ring padding), but INT8 quantization
    is applied *after* FP32 training (calibrate observers, then convert) rather
    than simulated during a fine-tuning phase. Reuses the entire PTQ pipeline from
    EmagerCNNQuantizedPTQ (FP32 train -> fold classifier BN -> fuse Conv+BN+ReLU
    -> calibrate -> convert -> CPU eval). Only two things change vs that class,
    exactly mirroring how EmagerCNNRingStridedQAT specializes EmagerCNNQuantizedQAT:
      - __init__ builds the RingStrided conv stack (RingPad2d interleaved with
        strided Conv2d) instead of the Base stack, plus the quant/dequant stubs.
      - _fuse_groups points at the Conv+BN+ReLU indices in *this* features layout.
        RingPad2d sits at indices 0/4/8, so the triplets are at 1-3, 5-7, 9-11.

    RingPad2d is quantization-safe (it wraps W with torch.cat, not F.pad's
    circular mode, which raises on quantized tensors). Useful comparisons:
      - EmagerCNNRingStrided    (same arch, FP32) — the pure quantization cost
      - EmagerCNNRingStridedQAT (same arch, QAT)  — PTQ vs QAT on this arch
      - EmagerCNNQuantizedPTQ   (Base arch, PTQ)  — architecture effect at equal precision

    input (B, H*W)
      |- BN1d                                   (kept FP32 -- raw signal range)
      |- reshape (B, 1, 4, 16)
      |- QuantStub                              (FP32 -> INT8 from here)
      |- RingPad(w=1,h=1) -> [Conv(1 ->32, 3x3, s=2) + BN + ReLU]  fused -> (B,32,2,8)
      |- RingPad(w=1,h=1) -> [Conv(32->32, 3x3, s=2) + BN + ReLU]  fused -> (B,32,1,4)
      |- RingPad(w=2,h=2) -> [Conv(32->32, 5x5, s=1) + BN + ReLU]  fused -> (B,32,1,4)
      |- Flatten -> (B, 128)
      |- Linear(128 -> 64)
      |- Dropout(0.5)
      |- [BN1d + ReLU]                          (BN1d folded into Linear before convert)
      |- Linear(64 -> C)
      |- DeQuantStub                            (INT8 -> FP32 out)
    """
    # RingPad2d occupies indices 0/4/8; Conv+BN+ReLU triplets are at 1-3, 5-7, 9-11.
    _fuse_groups = [["1", "2", "3"], ["5", "6", "7"], ["9", "10", "11"]]

    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        # Reuse the quantized scaffold from the parent (hparams, normalize, quant/
        # dequant stubs, criterion, _quantized), then replace the Base conv stack
        # and classifier with the RingStrided architecture.
        super().__init__(input_shape, num_classes, lr)

        # spatial size after two stride-2 convs (kernel=3, pad 1 each side via RingPad)
        h = (input_shape[0] + 2 - 3) // 2 + 1
        h = (h              + 2 - 3) // 2 + 1
        w = (input_shape[1] + 2 - 3) // 2 + 1
        w = (w              + 2 - 3) // 2 + 1
        flat = 32 * h * w   # 128 for the default (4, 16) input

        self.features   = nn.Sequential(
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            RingPad2d(pad_w=2, pad_h=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(flat, 64), nn.Dropout(0.5), nn.BatchNorm1d(64), nn.ReLU(inplace=False),
            nn.Linear(64, num_classes),
        )


class EmagerCNNRingStridedQAT(EmagerCNNQuantizedQAT):
    """
    EmagerCNNRingStrided architecture trained with Quantization-Aware Training.

    Stacks all three optimization axes used in this file:
      - strided spatial collapse (EmagerCNNStrided) — flatten drops 2048 -> 128,
        removing the dominant Linear cost
      - column-only ring padding (EmagerCNNRingStrided) — wraps the W axis (the
        physical electrode ring) and zero-pads H
      - INT8 QAT (EmagerCNNQuantizedQAT) — fake-quant during fine-tuning so the
        weights adapt to INT8 error

    Reuses the entire QAT pipeline from EmagerCNNQuantizedQAT (FP32 warm start ->
    fold classifier BN -> fuse_modules_qat -> prepare_qat -> fine-tune -> convert
    -> CPU eval). Only two things change vs that class:
      - __init__ builds the RingStrided conv stack (RingPad2d interleaved with
        strided Conv2d) instead of the Base stack, plus the quant/dequant stubs.
      - _fuse_groups points at the Conv+BN+ReLU indices in *this* features layout.
        RingPad2d sits at indices 0/4/8, so the triplets are at 1-3, 5-7, 9-11.

    RingPad2d is quantization-safe (it wraps W with torch.cat, not F.pad's
    circular mode, which raises on quantized tensors). Compare against
    EmagerCNNRingStrided (same arch, FP32) for the QAT accuracy/size tradeoff.

    input (B, H*W)
      |- BN1d                                   (kept FP32 -- raw signal range)
      |- reshape (B, 1, 4, 16)
      |- QuantStub                              (FP32 -> INT8 from here)
      |- RingPad(w=1,h=1) -> [Conv(1 ->32, 3x3, s=2) + BN + ReLU]  fused -> (B,32,2,8)
      |- RingPad(w=1,h=1) -> [Conv(32->32, 3x3, s=2) + BN + ReLU]  fused -> (B,32,1,4)
      |- RingPad(w=2,h=2) -> [Conv(32->32, 5x5, s=1) + BN + ReLU]  fused -> (B,32,1,4)
      |- Flatten -> (B, 128)
      |- Linear(128 -> 64)
      |- Dropout(0.5)
      |- [BN1d + ReLU]                          (BN1d folded into Linear at convert)
      |- Linear(64 -> C)
      |- DeQuantStub                            (INT8 -> FP32 out)
    """
    # RingPad2d occupies indices 0/4/8; Conv+BN+ReLU triplets are at 1-3, 5-7, 9-11.
    _fuse_groups = [["1", "2", "3"], ["5", "6", "7"], ["9", "10", "11"]]

    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        # Reuse the quantized scaffold from the parent (hparams, normalize, quant/
        # dequant stubs, criterion, _quantized), then replace the Base conv stack
        # and classifier with the RingStrided architecture.
        super().__init__(input_shape, num_classes, lr)

        # spatial size after two stride-2 convs (kernel=3, pad 1 each side via RingPad)
        h = (input_shape[0] + 2 - 3) // 2 + 1
        h = (h              + 2 - 3) // 2 + 1
        w = (input_shape[1] + 2 - 3) // 2 + 1
        w = (w              + 2 - 3) // 2 + 1
        flat = 32 * h * w   # 128 for the default (4, 16) input

        self.features   = nn.Sequential(
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            RingPad2d(pad_w=2, pad_h=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(flat, 64), nn.Dropout(0.5), nn.BatchNorm1d(64), nn.ReLU(inplace=False),
            nn.Linear(64, num_classes),
        )


# ==============================================================================
#  Few-shot prototypical models
#  Prototypes are COMPUTED as the mean embedding of a support set (gradient-free),
#  not learned by gradient. Models the deployment flow: train the embedding
#  offline, then calibrate on device from a few examples per gesture.
# ==============================================================================

class _EmagerProtoBase(_EmagerBase):
    """
    Few-shot prototypical classifier base (abstract — do not register directly).

    The conv stack produces a D-dim embedding; classification is by distance to
    one prototype per class, where each prototype is the MEAN embedding of a
    support set (computed, not learned). This is the Prototypical Networks
    inference rule (Snell et al. 2017) with prototypes obtained from a support
    set rather than a learnable nn.Parameter.

    Deployment flow modeled here:
      - offline (server)  : train the embedding f_θ on the training reps.
                            *How* f_θ is trained is the only thing subclasses
                            change (episodic vs cross-entropy pretrain).
      - on device         : the user records `n_shots` examples / gesture →
                            prototypes = mean of their embeddings. No gradient;
                            this is the "few-shot finetuning".
      - inference         : nearest prototype (softmax over -‖e - p_c‖²).

    fit() reports two accuracies on the SAME held-out query set:
      - BEFORE few-shot : prototypes computed from the TRAINING reps (generic /
                          "factory" prototypes shipped with the model).
      - AFTER  few-shot : prototypes computed from an n_shots support set drawn
                          from the held-out rep (on-device calibration).
    The held-out rep is split into `n_shots` support examples per class plus a
    query set; both accuracies use that same query set so the delta is fair.
    Both are logged; the AFTER-few-shot accuracy is returned as `test_acc`
    (the deployment number that lands in the benchmark table).

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ Conv2d(1  -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 3x3) + BN2d + ReLU
      └─ Conv2d(32 -> 32, 5x5) + BN2d + ReLU
      └─ Flatten                              →  (B, 32·H·W)
      └─ Linear(32·H·W -> D) + BN1d + ReLU    →  (B, D)  embedding
      └─ distance to prototypes:  logit_c = -‖e - p_c‖²/D  →  (B, C)
    """
    n_shots = 5   # support examples per class (training episodes AND on-device calibration)

    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3, embed_dim: int = 64):
        super().__init__()
        self.save_hyperparameters()
        n = int(np.prod(input_shape))

        self.normalize  = nn.BatchNorm1d(n)
        self.features   = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * n, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True),
        )
        # Prototypes are a buffer (state, not a learnable Parameter): set by
        # _prototypes_from() inside fit(), and saved/moved with the module.
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self.criterion  = nn.CrossEntropyLoss()

    # -- embedding + prototype-distance forward (inference / predict) ----------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x.view(x.size(0), -1))
        x = x.view(x.size(0), 1, *self.hparams.input_shape)
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        dist = (e.unsqueeze(1) - self.prototypes.unsqueeze(0)).pow(2).sum(dim=-1)
        return -dist / e.size(1)

    # -- gradient-free prototype computation (the "few-shot" step) -------------
    @torch.no_grad()
    def _prototypes_from(self, batches) -> torch.Tensor:
        """Mean embedding per class over an iterable of (x, y) batches."""
        self.eval()
        C, D = self.hparams.num_classes, self.hparams.embed_dim
        psum = torch.zeros(C, D, device=self.device)
        pcnt = torch.zeros(C, 1, device=self.device)
        for x, y in batches:
            e = self.embed(self.convert_input(x))
            y = y.to(self.device)
            for c in range(C):
                mask = y == c
                if mask.any():
                    psum[c] += e[mask].sum(dim=0)
                    pcnt[c] += mask.sum()
        return psum / pcnt.clamp(min=1)   # classes absent from the support stay at 0

    # -- split the held-out rep into an n_shots support set + a query set ------
    @staticmethod
    def _gather(dl):
        xs, ys = zip(*[(x, y) for x, y in dl])
        return torch.cat(xs), torch.cat(ys)

    @staticmethod
    def _split_support_query(x, y, k):
        s_idx, q_idx = [], []
        for c in y.unique():
            ci   = (y == c).nonzero(as_tuple=True)[0]
            perm = ci[torch.randperm(ci.numel())]
            s_idx.append(perm[:k])
            q_idx.append(perm[k:])
        s, q = torch.cat(s_idx), torch.cat(q_idx)
        return (x[s], y[s]), (x[q], y[q])

    # -- eval on the query set with whatever prototypes are currently set ------
    @torch.no_grad()
    def _eval(self, x, y, batch: int = 256):
        self.eval()
        correct, total, loss_sum = 0, 0, 0.0
        for i in range(0, x.size(0), batch):
            logits = self(self.convert_input(x[i:i + batch]))
            yb     = y[i:i + batch].to(self.device)
            loss_sum += self.criterion(logits, yb).item() * yb.size(0)
            correct  += (logits.argmax(dim=1) == yb).sum().item()
            total    += yb.size(0)
        return correct / total, loss_sum / total

    # -- held-out rep → (support, query) --------------------------------------
    def _split_fewshot(self, test_dataloader):
        """
        Held-out rep = "new session": carve out n_shots support examples per class
        (the on-device calibration set) and keep the rest as the query set.

        Split out of _eval_fewshot so the quantized subclasses can get at the support
        set *before* they quantize — see EmagerCNNProtoRingStridedPTQ.calib_source.
        """
        x, y = self._gather(test_dataloader)
        return self._split_support_query(x, y, self.n_shots)

    # -- generic protos → few-shot protos → report ----------------------------
    def _eval_fewshot(self, train_dataloader, test_dataloader, proto_model=None, split=None):
        """
        Calibrate + score, given an already-trained embedding. Split out of fit()
        so the quantized subclasses can run it *after* converting to INT8 — their
        prototypes must come from the quantized embedding, which is the one the
        device actually runs.

        proto_model: embedding used to COMPUTE the prototypes; defaults to self.
            The quantized subclasses pass a pre-quantization copy here to run the
            `proto_precision="fp32"` ablation, while classification still happens
            through self (the INT8 model).
        split: precomputed (support, query) from _split_fewshot, when the caller
            already needed it earlier (e.g. to calibrate observers on the support set).
        """
        if test_dataloader is None:
            return

        proto_model = proto_model if proto_model is not None else self
        (sx, sy), (qx, qy) = split if split is not None else self._split_fewshot(test_dataloader)

        # Stage B: BEFORE few-shot — generic prototypes from the training reps.
        self.prototypes = proto_model._prototypes_from(train_dataloader)
        self.acc_before_fewshot, _ = self._eval(qx, qy)

        # Stage C: AFTER few-shot — prototypes from the n_shots support set.
        self.prototypes = proto_model._prototypes_from([(sx, sy)])
        self.acc_after_fewshot, loss_after = self._eval(qx, qy)

        logger.info(f"  [proto] before few-shot (generic protos): acc={self.acc_before_fewshot:.1%}")
        logger.info(
            f"  [proto] after  {self.n_shots}-shot calibration:   acc={self.acc_after_fewshot:.1%}  "
            f"(delta {self.acc_after_fewshot - self.acc_before_fewshot:+.1%})"
        )
        return [{"test_acc": self.acc_after_fewshot, "test_loss": loss_after}]

    # -- offline-train embedding → generic protos → few-shot protos → report --
    def fit(self, train_dataloader, test_dataloader=None, max_epochs: int = 10):
        # Stage 1: offline embedding training. The loss lives in _shared_step,
        # which each subclass overrides (episodic vs cross-entropy pretrain).
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        trainer.fit(self, train_dataloader)

        # Stage 2: calibrate on the held-out rep and report before/after.
        return self._eval_fewshot(train_dataloader, test_dataloader)


class EmagerCNNProtoEpisodic(_EmagerProtoBase):
    """
    Episodic Prototypical Network (faithful — Snell et al. 2017).

    The embedding is trained the same way it is used at deployment: every
    mini-batch is treated as an episode. Within the batch, each class's samples
    are split into a few support examples (mean → prototype) and the rest as
    query; the query is classified by distance to those batch prototypes and the
    cross-entropy loss is backpropagated. The embedding is thus directly
    optimized for the gradient-free, mean-prototype rule used on device.

    Only _shared_step differs from the base; everything else (architecture,
    the generic→few-shot eval in fit, predict) is inherited.
    """
    def _shared_step(self, batch):
        x, y = batch
        e = self.embed(x)

        protos, q_e, q_t, pos = [], [], [], 0
        for c in y.unique():
            ci = (y == c).nonzero(as_tuple=True)[0]
            if ci.numel() < 2:                       # need ≥1 support and ≥1 query
                continue
            perm = ci[torch.randperm(ci.numel(), device=e.device)]
            n_s  = max(1, min(self.n_shots, ci.numel() - 1))
            protos.append(e[perm[:n_s]].mean(dim=0))
            q_e.append(e[perm[n_s:]])
            q_t.append(torch.full((perm.numel() - n_s,), pos, dtype=torch.long, device=e.device))
            pos += 1

        if pos < 2:                                  # unusable episode (rare; e.g. tiny batch)
            return e.sum() * 0.0, torch.zeros((), device=e.device)

        protos = torch.stack(protos)                 # (P, D)
        q_e    = torch.cat(q_e)                      # (Q, D)
        q_t    = torch.cat(q_t)                      # (Q,)  episode-local class indices
        dist   = (q_e.unsqueeze(1) - protos.unsqueeze(0)).pow(2).sum(dim=-1)
        logits = -dist / q_e.size(1)
        loss   = self.criterion(logits, q_t)
        acc    = (logits.argmax(dim=1) == q_t).float().mean()
        return loss, acc


class EmagerCNNProtoCE(_EmagerProtoBase):
    """
    Cross-entropy pretrained embedding ("baseline" few-shot — Chen et al. 2019).

    The embedding is trained with an ordinary Linear head + cross-entropy
    (standard classification). After training the head is dropped, and the same
    generic→few-shot prototype evaluation as the base is used. Simpler and often
    as accurate as episodic training, but the embedding is not optimized for the
    distance-based rule directly.

    The Linear head exists only during offline training; it is removed after
    fit() so the deployed model is just the embedding + computed prototypes.
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3, embed_dim: int = 64):
        super().__init__(input_shape, num_classes, lr, embed_dim)
        self._ce_head = nn.Linear(embed_dim, num_classes)   # offline-training head only

    def _shared_step(self, batch):
        x, y   = batch
        logits = self._ce_head(self.embed(x))
        loss   = self.criterion(logits, y)
        acc    = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def fit(self, train_dataloader, test_dataloader=None, max_epochs: int = 10):
        out = super().fit(train_dataloader, test_dataloader, max_epochs)
        self._ce_head = None   # deployed model = embedding + prototypes, no CE head
        return out


# ==============================================================================
#  Few-shot prototypical models on the RingStrided architecture (deployment line)
#  Same prototype rule as above, but on the smallest architecture in the file,
#  with INT8 leaves. This is the MCU-targeted branch.
# ==============================================================================

class EmagerCNNProtoRingStrided(EmagerCNNProtoEpisodic):
    """
    Episodic prototypical network on the EmagerCNNRingStrided backbone.

    Combines the two directions that each paid off on their own axis:
      - RingStrided (strided spatial collapse + column-only ring padding) — the
        smallest architecture in this file, ~12x fewer params than Base.
      - Episodic prototypes — classification by distance to prototypes computed
        from a support set, so per-session calibration needs no on-device backprop.

    Only __init__ differs from EmagerCNNProtoEpisodic: the embedding backbone is
    the RingStrided conv stack instead of the Base stack. The episodic loss
    (_shared_step), the prototype computation and the generic->few-shot eval in
    fit() are all inherited unchanged.

    Because the strided stack collapses (4,16) -> (1,4) before flatten, the
    embedding head is Linear(128 -> 64) rather than Linear(2048 -> 64) — which is
    where nearly all of EmagerCNNProtoEpisodic's 167K params live.

    input (B, H*W)
      |- BN1d
      |- reshape (B, 1, 4, 16)
      |- RingPad(w=1,h=1) -> Conv2d(1 ->32, 3x3, s=2, p=0) + BN2d + ReLU  -> (B,32,2,8)
      |- RingPad(w=1,h=1) -> Conv2d(32->32, 3x3, s=2, p=0) + BN2d + ReLU  -> (B,32,1,4)
      |- RingPad(w=2,h=2) -> Conv2d(32->32, 5x5, s=1, p=0) + BN2d + ReLU  -> (B,32,1,4)
      |- Flatten                            -> (B, 128)
      |- Linear(128 -> D) + BN1d + ReLU     -> (B, D)  embedding
      |- distance to prototypes: logit_c = -||e - p_c||^2/D  -> (B, C)
    """
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3, embed_dim: int = 64):
        super().__init__(input_shape, num_classes, lr, embed_dim)

        # spatial size after two stride-2 convs (kernel=3, pad 1 each side via RingPad)
        h = (input_shape[0] + 2 - 3) // 2 + 1
        h = (h              + 2 - 3) // 2 + 1
        w = (input_shape[1] + 2 - 3) // 2 + 1
        w = (w              + 2 - 3) // 2 + 1
        flat = 32 * h * w   # 128 for the default (4, 16) input

        # ReLU(inplace=False) throughout: the PTQ/QAT subclasses fuse this exact
        # stack, and torch.ao.quantization's fusion requires out-of-place ReLU.
        # Declaring it that way once here lets them inherit the architecture
        # rather than restate it.
        self.features = nn.Sequential(
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            RingPad2d(pad_w=1, pad_h=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            RingPad2d(pad_w=2, pad_h=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear(flat, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=False),
        )


class EmagerCNNProtoRingStridedPTQ(EmagerCNNProtoRingStrided):
    """
    EmagerCNNProtoRingStrided with the embedding quantized to INT8 after training (PTQ).

    What is and isn't quantized
    ---------------------------
    The QuantStub/DeQuantStub bracket the *embedding* only; the prototype distance
    stays FP32:

        BN1d -> QuantStub -> [ INT8 conv stack + INT8 Linear(128->D) ] -> DeQuantStub
             -> ||e - p_c||^2 in FP32

    That is deliberate, not a shortcut. The distance is C*D (7*64 = 448) MACs against
    ~152K in the conv stack, so quantizing it would buy nothing measurable, and eager
    mode has no quantized kernel for the broadcast-subtract/pow it needs. It also
    mirrors how this runs on an MCU: an INT8 kernel (CMSIS-NN / TFLite Micro) for the
    conv stack, a few hundred float ops for the nearest-prototype lookup.

    Why prototypes are computed AFTER convert()
    -------------------------------------------
    fit() trains FP32, quantizes, and only *then* computes prototypes — through the
    INT8 embedding. This is what the device does: it ships with a frozen INT8 network
    and averages the embeddings *that network* produces for the user's calibration
    examples. Computing prototypes from the FP32 embedding and then quantizing would
    put the prototypes in a slightly different space than the embeddings they are
    compared against at inference.

    A pleasant consequence: the calibration step is inherently quantization-aware
    with no extra machinery, since the prototypes are means of quantized embeddings.

    Pipeline (inside fit()):
      1. FP32 episodic training of the embedding (inherited _shared_step)
      2. Fold the embedding head's BatchNorm1d into its Linear (exact in eval mode;
         QuantizedCPU has no BN1d kernel)
      3. Fuse Conv+BN+ReLU triplets in the conv stack
      4. Calibrate observers on ~10 train batches (through embed(), the quantized region)
      5. convert() to INT8, then compute prototypes and report before/after few-shot

    Backend defaults to fbgemm (x86 host); set qbackend = "qnnpack" for ARM.

    Strategy knobs (`proto_precision`, `calib_source`)
    -------------------------------------------------
    A prototypical model has a *second*, gradient-free training step — computing the
    prototypes — that a plain classifier does not. That creates timing questions with
    no equivalent in the PTQ-vs-QAT axis, and they are cheap to test because no
    backprop is involved. Both knobs default to the deployment-faithful choice; the
    non-default settings exist to measure whether that choice actually matters. See
    the two subclasses at the end of this file for the named ablation variants.
    """
    qbackend       = "fbgemm"   # "qnnpack" for ARM (e.g. STM32 / Cortex-M) deployment
    _calib_batches = 10
    # RingPad2d occupies indices 0/4/8; Conv+BN+ReLU triplets are at 1-3, 5-7, 9-11.
    _fuse_groups   = [["1", "2", "3"], ["5", "6", "7"], ["9", "10", "11"]]

    # Which embedding computes the prototypes.
    #   "int8" (default) — the quantized embedding, i.e. prototypes are means of the
    #       same INT8 vectors they will be compared against. This is what the device
    #       does; it has no other network.
    #   "fp32"  — the pre-quantization embedding. NOT deployable for the on-device
    #       calibration step (the device has only the INT8 net), so this is an
    #       ablation, not a shippable strategy: it measures what the prototype/embedding
    #       space mismatch actually costs. If the two score the same, the ordering
    #       inside fit() does not matter and the naive order is fine.
    proto_precision = "int8"    # "int8" | "fp32"

    # What the activation observers calibrate on.
    #   "train" (default) — batches from the training sessions, at the factory.
    #   "support" — the held-out session's support set, i.e. the user's own k shots.
    #       A diagnostic rather than a shippable strategy: TFLite Micro bakes scales
    #       into the flatbuffer at export, so an MCU cannot re-run observers. What it
    #       tells you is whether train-set activation ranges transfer to a new session
    #       — if support-calibrated INT8 is much better, the factory calibration set is
    #       too narrow and should be widened.
    calib_source    = "train"   # "train" | "support"

    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3, embed_dim: int = 64):
        super().__init__(input_shape, num_classes, lr, embed_dim)
        if self.proto_precision not in ("int8", "fp32"):
            raise ValueError(f"proto_precision must be 'int8' or 'fp32', got {self.proto_precision!r}")
        if self.calib_source not in ("train", "support"):
            raise ValueError(f"calib_source must be 'train' or 'support', got {self.calib_source!r}")
        self.quant      = torch.ao.quantization.QuantStub()
        self.dequant    = torch.ao.quantization.DeQuantStub()
        self._quantized = False

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x.view(x.size(0), -1))
        x = x.view(x.size(0), 1, *self.hparams.input_shape)
        x = self.quant(x)
        x = self.features(x)
        return self.dequant(x)     # prototype distance runs in FP32 on the result

    def fit(self, train_dataloader, test_dataloader=None, max_epochs: int = 10):
        # Stage 1: offline FP32 embedding training (episodic loss)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        trainer.fit(self, train_dataloader)

        # Stages 2-4
        return self.quantize_from_pretrained(train_dataloader, test_dataloader)

    def quantize_from_pretrained(self, train_dataloader, test_dataloader=None, calib_loader=None):
        """
        Stages 2-4: quantize an **already-trained** FP32 embedding, then compute
        prototypes and report before/after few-shot.

        Public, and split out of fit(), because this class's FP32 stage-1 weights are
        bit-identical to `EmagerCNNProtoRingStrided`'s for the same seed: same backbone,
        same episodic loss, same data order, and the Quant/DeQuant stubs hold no
        parameters so they draw no RNG during init and are no-ops in FP32. A benchmark
        harness can therefore train stage 1 **once** per (fold, seed), load the weights
        into each quantized variant, and call this — turning N trainings into 1 with no
        approximation. See STAGE1_SOURCE in examples/training/eval_fewshot_loso.py.

        This matters beyond speed: a converted model cannot be round-tripped. Pickling
        one and loading it back yields fused modules with no `_modules` dict (even
        .eval() raises), and its state_dict will not load into a fresh FP32 instance
        because convert() restructured the module tree. Deriving from FP32 weights is
        the supported way to reproduce a quantized model.

        QAT overrides this with its own pipeline, so callers can treat it polymorphically.

        calib_loader: explicit observer-calibration batches. When given, `calib_source`
            is not consulted — the caller has already decided what to calibrate on. This
            is how a harness that owns the support/query split itself feeds the support
            set in without handing over a test_dataloader.
        """
        # Both strategy knobs have to be settled BEFORE convert(): the "fp32" prototype
        # source needs a copy of the un-quantized embedding, and "support" calibration
        # needs the held-out split in hand.
        proto_model = copy.deepcopy(self) if self.proto_precision == "fp32" else None
        split       = self._split_fewshot(test_dataloader) if test_dataloader is not None else None
        if calib_loader is None:
            calib_loader = train_dataloader
            if self.calib_source == "support":
                if split is None:
                    raise ValueError(
                        "calib_source='support' needs either a test_dataloader to draw the "
                        "support set from, or an explicit calib_loader"
                    )
                (sx, sy), _ = split
                calib_loader = [(sx, sy)]

        # Quantize the embedding to INT8 (runs even without a test loader, so a fit()
        # used purely to produce a deployable model still converts)
        self._apply_ptq(calib_loader)

        # Prototypes + before/after report. Classification always runs through self
        # (the INT8 model); only the prototype *source* varies.
        return self._eval_fewshot(train_dataloader, test_dataloader,
                                  proto_model=proto_model, split=split)

    def _apply_ptq(self, calib_loader):
        import torch.ao.quantization as taq

        self.eval()
        self.cpu()

        self._fold_embed_bn()
        taq.fuse_modules(self.features, self._fuse_groups, inplace=True)

        self.qconfig = taq.get_default_qconfig(self.qbackend)
        taq.prepare(self, inplace=True)

        # Calibration goes through embed(), not forward(): embed() is exactly the
        # quantized region, and forward() would additionally hit the prototype
        # distance while the prototype buffer is still zeros.
        with torch.no_grad():
            for i, (x, _) in enumerate(calib_loader):
                self.embed(x.cpu())
                if i + 1 >= self._calib_batches:
                    break

        taq.convert(self, inplace=True)
        self._quantized = True

    def _fold_embed_bn(self):
        """
        Fold the embedding head's BatchNorm1d into its preceding Linear.

            Before:  [..., Flatten, Linear(flat, D), BN1d(D), ReLU]
            After:   [..., Flatten, Linear_folded,            ReLU]

        Addressed from the tail so it stays correct regardless of how many layers
        the conv stack has. Leaves _fuse_groups' indices (all < flatten) untouched.
        """
        layers        = list(self.features)
        lin, bn, relu = layers[-3], layers[-2], layers[-1]
        self.features = nn.Sequential(*layers[:-3], _fold_bn1d_into_linear(lin, bn), relu)

    def convert_input(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x.float().cpu() if self._quantized else super().convert_input(x)


class EmagerCNNProtoRingStridedQAT(EmagerCNNProtoRingStridedPTQ):
    """
    Quantization-aware-training counterpart to EmagerCNNProtoRingStridedPTQ.

    Same architecture and same FP32-embedding/FP32-distance split; the only
    difference is that INT8 error is simulated during a fine-tuning phase so the
    embedding weights adapt to it, rather than being quantized in one shot after
    training. Inherits embed(), the BN fold, and convert_input from the PTQ class.

    Note the interaction worth watching here: the fine-tuning uses the *episodic*
    loss, so the embedding is adapting to fake-quant while being optimized for the
    mean-prototype rule. That is the closest this file gets to training a model for
    exactly what it does at deployment (INT8 embedding + computed prototypes).

    Pipeline (inside fit()):
      1. FP32 episodic warm start — also populates the BN running stats the fold needs
      2. Fold the embedding head's BatchNorm1d into its Linear (exact in eval mode)
      3. Fuse Conv+BN+ReLU with fuse_modules_qat -> ConvBnReLU2d (BN stays trainable)
      4. prepare_qat() inserts fake-quant + observers
      5. Fine-tune qat_epochs (default 5) on CPU, episodic loss, fake quant active
      6. convert() to real INT8, then compute prototypes and report before/after

    Compare against EmagerCNNProtoRingStridedPTQ for the QAT-vs-PTQ delta on this
    architecture, and against EmagerCNNProtoRingStrided for the pure cost of INT8.
    """
    qat_epochs = 5   # fine-tuning epochs after the FP32 warm start

    def _reject_ptq_knobs(self):
        """
        The PTQ strategy knobs do not carry over, so refuse them rather than silently
        ignore: QAT has no separate observer-calibration pass (ranges are learned during
        fine-tuning on train_dataloader, so calib_source is not a choice), and its
        pre-convert model is already fake-quantized, so "fp32" prototypes would not be
        FP32 in any meaningful sense.
        """
        if self.proto_precision != "int8" or self.calib_source != "train":
            raise ValueError(
                f"{type(self).__name__} does not support proto_precision="
                f"{self.proto_precision!r} / calib_source={self.calib_source!r}; "
                "those knobs are PTQ-only. Use EmagerCNNProtoRingStridedPTQ to vary them."
            )

    def fit(self, train_dataloader, test_dataloader=None, max_epochs: int = 10):
        self._reject_ptq_knobs()

        # Stage 1: FP32 episodic warm start
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        trainer.fit(self, train_dataloader)

        # Stages 2-6
        return self.quantize_from_pretrained(train_dataloader, test_dataloader)

    def quantize_from_pretrained(self, train_dataloader, test_dataloader=None, calib_loader=None):
        """
        Stages 2-6, assuming the FP32 warm start is already done (weights loaded).

        Overrides the PTQ pipeline with the QAT one, so a harness can call this
        polymorphically on either class. The warm start this replaces is bit-identical
        to `EmagerCNNProtoRingStrided`'s training, so QAT can share the same cached FP32
        stage-1 as the PTQ variants and pay only the fine-tuning (see STAGE1_SOURCE).
        """
        import torch.ao.quantization as taq

        self._reject_ptq_knobs()
        if calib_loader is not None:
            raise ValueError(
                f"{type(self).__name__} learns activation ranges during QAT fine-tuning; "
                "an explicit calib_loader does not apply (that is a PTQ concept)."
            )

        # Stages 2-4: fold embedding BN, fuse for QAT, insert fake-quant (on CPU)
        self._prepare_qat()

        # Stage 5: QAT fine-tuning with fake quant active. Forced onto CPU so the
        # fbgemm/qnnpack fake-quant + convert path is consistent on any host.
        qat_trainer = L.Trainer(
            max_epochs=self.qat_epochs,
            accelerator="cpu",
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        qat_trainer.fit(self, train_dataloader)

        # Stage 6: convert to real INT8, then prototypes + before/after report
        self.eval()
        taq.convert(self, inplace=True)
        self._quantized = True
        return self._eval_fewshot(train_dataloader, test_dataloader)

    def _prepare_qat(self):
        import torch.ao.quantization as taq

        self.eval()
        self.cpu()
        # Fold now that the FP32 warm start has given BN meaningful running stats.
        self._fold_embed_bn()

        # fuse_modules_qat keeps BN inside ConvBnReLU2d so it stays trainable during
        # fine-tuning (plain fuse_modules would fold it away immediately — correct
        # for PTQ, but it defeats QAT).
        self.train()
        taq.fuse_modules_qat(self.features, self._fuse_groups, inplace=True)
        self.qconfig = taq.get_default_qat_qconfig(self.qbackend)
        taq.prepare_qat(self, inplace=True)


# ==============================================================================
#  Quantization-timing ablations (PTQ only)
#  Named variants of EmagerCNNProtoRingStridedPTQ that flip one strategy knob each,
#  so the benchmark harness can select them by name. Neither is a shippable
#  strategy -- both exist to measure whether the default was the right call.
# ==============================================================================

class EmagerCNNProtoRingStridedPTQFp32Protos(EmagerCNNProtoRingStridedPTQ):
    """
    Ablation: compute prototypes from the **FP32** embedding instead of the INT8 one.

    Everything else matches `EmagerCNNProtoRingStridedPTQ` — classification still runs
    through the INT8 embedding. Only the prototype source changes: fit() keeps a copy of
    the un-quantized model and computes both the generic and the k-shot prototypes
    through it.

    **This is not deployable**, and that is the point. On device there is only the INT8
    network, so on-device calibration cannot produce FP32 prototypes. The variant exists
    to isolate one question the default answers by assumption:

        Does it matter that prototypes are means of the *same* INT8 vectors they are
        later compared against?

    Read it against `EmagerCNNProtoRingStridedPTQ`:
      - **Same accuracy** -> the space mismatch is negligible, the ordering inside fit()
        is not load-bearing, and the naive order would have been fine.
      - **INT8 protos better** -> computing prototypes after convert() is doing real
        work, and the default is correct.
      - **FP32 protos better** -> INT8 embedding noise hurts the prototype means more
        than the mismatch costs. That would argue for shipping FP32-derived *generic*
        prototypes from the factory (legal — those are computed at the factory), even
        though on-device recalibration must stay INT8.
    """
    proto_precision = "fp32"


class EmagerCNNProtoRingStridedPTQSupportCalib(EmagerCNNProtoRingStridedPTQ):
    """
    Diagnostic: calibrate the activation observers on the held-out session's **support
    set** (the user's own k shots) instead of on the training sessions.

    Everything else matches `EmagerCNNProtoRingStridedPTQ`, prototypes included (still
    INT8). Only the data the observers see while picking activation scales/zero-points
    changes.

    **Not directly shippable**: TFLite Micro bakes quantization scales into the
    flatbuffer at export time, so an MCU cannot re-run observers against user data. What
    this measures is whether factory-calibrated activation ranges *transfer* to a new
    session:

      - **Same accuracy** -> train-session ranges cover a new session fine; nothing to do.
      - **Support-calibrated better** -> the factory calibration set is too narrow. That
        is actionable without any on-device work: widen the calibration set (more
        sessions / subjects) or switch to a percentile/histogram observer — both are
        export-time changes.

    Requires a test_dataloader in fit(); there is no support set without one.
    """
    calib_source = "support"
