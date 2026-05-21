import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


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
        self.eval()
        with torch.no_grad():
            return F.softmax(self(self.convert_input(x)), dim=1).cpu().numpy()

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
    Circular padding on all conv layers — treats the column axis (electrodes) as periodic.
    The 16 electrode columns sit on a physical ring, so the last column is spatially
    adjacent to the first. Circular padding reflects that instead of padding with zeros.

    Architecture is identical to Base — only the padding mode changes.

    input (B, H*W)
      └─ BN1d
      └─ reshape (B, 1, H, W)
      └─ Conv2d(1  -> 32, 3x3, padding_mode='circular') + BN2d + ReLU
      └─ Conv2d(32 -> 32, 3x3, padding_mode='circular') + BN2d + ReLU
      └─ Conv2d(32 -> 32, 5x5, padding_mode='circular') + BN2d + ReLU
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
            nn.Conv2d(1,  32, kernel_size=3, padding=1, padding_mode="circular"), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode="circular"), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, padding_mode="circular"), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * n, 256), nn.Dropout(0.5), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
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
