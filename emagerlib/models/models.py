import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from brevitas import quant
import brevitas.nn as qnn

from sklearn.metrics import accuracy_score


class EmagerCNN(L.LightningModule):
    def __init__(self, input_shape, num_classes, quantization=-1):
        """
        Create a reference EmagerCNN model.

        Parameters:
            - input_shape: shape of input data
            - num_classes: number of classes
            - quantization: bit-width of weights and activations. >=32 or <0 for no quantization
        """
        super().__init__()

        self.input_shape = input_shape
        self.normalize = nn.BatchNorm1d(np.prod(input_shape))

        output_sizes = [32, 32, 32, 256]

        self.loss = nn.CrossEntropyLoss()

        self.bn1 = nn.BatchNorm2d(output_sizes[0])
        self.bn2 = nn.BatchNorm2d(output_sizes[1])
        self.bn3 = nn.BatchNorm2d(output_sizes[2])
        self.flat = nn.Flatten()
        self.dropout4 = nn.Dropout(0.5)
        self.bn4 = nn.BatchNorm1d(output_sizes[3])

        if quantization < 0 or quantization >= 32:
            self.inp = nn.Identity()
            self.conv1 = nn.Conv2d(1, output_sizes[0], 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(output_sizes[1], output_sizes[2], 5, padding=2)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(
                output_sizes[2] * np.prod(self.input_shape),
                output_sizes[3],
            )
            self.relu4 = nn.ReLU()
            self.fc5 = nn.Linear(output_sizes[3], num_classes)
        else:
            # FINN 0.10: QuantConv2d MUST have bias=False !!
            self.inp = qnn.QuantIdentity()
            self.conv1 = qnn.QuantConv2d(
                1,
                output_sizes[0],
                3,
                padding=1,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu1 = qnn.QuantReLU(bit_width=quantization)
            self.conv2 = qnn.QuantConv2d(
                output_sizes[0],
                output_sizes[1],
                3,
                padding=1,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu2 = qnn.QuantReLU(bit_width=quantization)
            self.conv3 = qnn.QuantConv2d(
                output_sizes[1],
                output_sizes[2],
                3,
                padding=1,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu3 = qnn.QuantReLU(bit_width=quantization)
            self.fc4 = qnn.QuantLinear(
                output_sizes[2] * np.prod(self.input_shape),
                output_sizes[3],
                bias=True,
                weight_bit_width=quantization,
            )
            self.relu4 = qnn.QuantReLU(bit_width=quantization)
            self.fc5 = qnn.QuantLinear(
                output_sizes[3],
                num_classes,
                bias=True,
                weight_bit_width=quantization,
            )

    def forward(self, x):
        x = self.normalize(x.view(x.size(0), -1))
        x = x.view(-1, 1, *self.input_shape)
        out = self.inp(x)
        out = self.bn1(self.relu1(self.conv1(out)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.flat(out)
        out = self.bn4(self.relu4(self.dropout4(self.fc4(out))))
        logits = self.fc5(out)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = self.loss(y, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = self.loss(y, y_true)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = self.loss(y, y_true)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return acc, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
    # ----- LibEMG -----

    def convert_input(self, x):
        """Convert arbitrary input to a Torch tensor

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x.type(torch.float32).to(self.device)

    def predict_proba(self, x):
        x = self.convert_input(x)
        with torch.no_grad():
            return F.softmax(self(x), dim=1).cpu().detach().numpy()

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def fit(self, train_dataloader, test_dataloader=None, max_epochs=10):

        self.train()
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        )
        trainer.fit(self, train_dataloader)
        res = None
        if test_dataloader is not None:
            res = trainer.test(self, test_dataloader)

        return res


class EmagerSCNN(L.LightningModule):
    def __init__(self, input_shape, quantization=-1):
        """
        Create a reference Siamese EmagerCNN model.

        Parameters:
            - input_shape: shape of input data
            - quantization: bit-width of weights and activations. >=32 or <0 for no quantization
        """
        super().__init__()

        # Test attributes
        self.test_preds = np.ndarray((0,), dtype=np.uint8)

        # Model definition
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.input_shape = input_shape

        output_sizes = [32, 32, 32, 256]

        self.bn1 = nn.BatchNorm2d(output_sizes[0])
        self.bn2 = nn.BatchNorm2d(output_sizes[1])
        self.bn3 = nn.BatchNorm2d(output_sizes[2])
        self.flat = nn.Flatten()

        if quantization < 0 or quantization >= 32:
            self.inp = nn.Identity()
            self.conv1 = nn.Conv2d(1, output_sizes[0], 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(output_sizes[1], output_sizes[2], 5, padding=2)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(output_sizes[2] * np.prod(self.input_shape))
        else:
            self.inp = qnn.QuantIdentity()
            self.conv1 = qnn.QuantConv2d(
                1,
                output_sizes[0],
                3,
                padding=1,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu1 = qnn.QuantReLU(bit_width=quantization)
            self.conv2 = qnn.QuantConv2d(
                output_sizes[0],
                output_sizes[1],
                3,
                padding=1,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu2 = qnn.QuantReLU(bit_width=quantization)
            self.conv3 = qnn.QuantConv2d(
                output_sizes[1],
                output_sizes[2],
                5,
                padding=2,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu3 = qnn.QuantReLU(bit_width=quantization)
            self.fc4 = qnn.QuantLinear(
                output_sizes[2] * np.prod(self.input_shape),
                output_sizes[3],
                bias=True,
                weight_bit_width=quantization,
                # bit_width=8,
                # input_quant=quant.Int8ActPerTensorFloat,
                # output_quant=quant.Int8ActPerTensorFloat,
            )

    def forward(self, x):
        out = torch.reshape(x, (-1, 1, *self.input_shape))
        out = self.inp(out)
        out = self.bn1(self.relu1(self.conv1(out)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.flat(out)
        out = self.fc4(out)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = self.loss(anchor, positive, negative)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = self.loss(anchor, positive, negative)
        self.log("val_loss", loss)
        return loss

    # def test_step(self, batch, batch_idx):
    #     if batch_idx == 0:
    #         self.test_preds = np.ndarray((0,), dtype=np.uint8)

    #     x, y_true = batch
    #     embeddings = self(x).cpu().detach().numpy()

    #     y = dp.cosine_similarity(embeddings, self.embeddings, True)
    #     y_true = y_true.cpu().detach().numpy()
    #     acc = accuracy_score(y_true, y, normalize=True)

    #     self.log("test_acc", acc)
    #     self.test_preds = np.concatenate((self.test_preds, y))
    #     return acc

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def set_target_embeddings(self, embeddings):
        self.embeddings = embeddings
