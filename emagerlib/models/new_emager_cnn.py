import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


# ==============================================================================
#  Padding primitives
# ==============================================================================

class RingPad2d(nn.Module):
    """
    Per-axis padding for the electrode grid:
      - circular on W (electrode columns form a physical ring: col 0 is adjacent to col 15)
      - zero on H (rows do not loop)

    Used by EmagerCNNCircular and EmagerCNNRingStrided. Conv2d after this layer should use padding=0.
    """
    def __init__(self, pad_w: int, pad_h: int):
        super().__init__()
        self.pad_w = pad_w
        self.pad_h = pad_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0.0)
        return x


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
        taq.fuse_modules(
            self.features,
            [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]],
            inplace=True,
        )

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

        scale = bn1d.weight / torch.sqrt(bn1d.running_var + bn1d.eps)
        shift = bn1d.bias - bn1d.running_mean * scale

        folded = nn.Linear(lin0.in_features, lin0.out_features, bias=True)
        bias0  = lin0.bias.data if lin0.bias is not None else torch.zeros(lin0.out_features)
        folded.weight.data = lin0.weight.data * scale.unsqueeze(1)
        folded.bias.data   = bias0 * scale + shift

        self.classifier = nn.Sequential(folded, dropout, relu, lin1)

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
        taq.fuse_modules_qat(
            self.features,
            [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]],
            inplace=True,
        )
        self.qconfig = taq.get_default_qat_qconfig(self.qbackend)
        taq.prepare_qat(self, inplace=True)


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
