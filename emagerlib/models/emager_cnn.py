import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


class EmagerCNN(L.LightningModule):
    def __init__(self, input_shape: tuple, num_classes: int, lr: float = 1e-3):
        """
        Args:
            input_shape: spatial shape of one sample, e.g. (4, 16)
            num_classes: number of gesture classes
            lr: learning rate for AdamW
        """
        super().__init__()
        self.save_hyperparameters()

        n_flat = int(np.prod(input_shape))

        self.normalize = nn.BatchNorm1d(n_flat)

        self.features = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * n_flat, 256),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x.view(x.size(0), -1))
        x = x.view(x.size(0), 1, *self.hparams.input_shape)
        return self.classifier(self.features(x))

    def _shared_step(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    # ----- LibEMG interface -----

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
