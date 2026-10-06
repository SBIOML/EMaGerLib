import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset


def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]


def make_data_loader(windows, classes, batch_size=64):
    obj = DL_input_data(windows, classes)
    dl = DataLoader(obj, batch_size=batch_size, shuffle=True)
    return dl


class MLP(nn.Module):
    def __init__(self, layers=[64, 64, 64]):
        super().__init__()
        self.layers = layers
        fix_random_seed(0)

    def setup(self, n_classes, n_features):
        self.dropout = nn.Dropout(p=0.2)
        self.initial_layer = nn.Linear(n_features, self.layers[0])
        self.layer1 = nn.Linear(self.layers[0], self.layers[1])
        self.layer2 = nn.Linear(self.layers[1], self.layers[2])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.layers[-1], n_classes)
        self.n_classes = n_classes

    def forward_once(self, out):
        out = self.initial_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

    def forward(self, x):
        out = self.forward_once(x)
        out = self.output_layer(out)
        return out

    def calibrate(self, features, labels, learning_rate=1e-3, num_epochs=100, verbose=False):
        """
        Calibrate the MLP on pre-computed features.

        Args:
            features: Pre-computed feature array (n_samples, n_features)
            labels: Class labels for each sample
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            verbose: Print progress
        """
        training_losses = []

        self.setup(len(np.unique(labels)), features.shape[1])
        tr_dl = make_data_loader(features, labels)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        loss_function = nn.CrossEntropyLoss()
        self.log = {"training_loss": [], "training_accuracy": []}
        epoch = 0
        while epoch < num_epochs:
            tr_loss = []
            self.train()
            for data, batch_labels in tr_dl:
                optimizer.zero_grad()
                data = data.to(device)
                batch_labels = batch_labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, batch_labels)
                loss.backward()
                optimizer.step()
                tr_loss.append(loss.item())

                acc = sum(torch.argmax(output, 1) == batch_labels) / batch_labels.shape[0]
                self.log["training_loss"] += [(epoch, loss.item())]
                self.log["training_accuracy"] += [(epoch, acc)]
            scheduler.step()
            training_losses.append(np.mean(tr_loss))
            if len(training_losses) > 50:
                has_improved = False
                for i in range(2, 6):
                    if training_losses[-i] - training_losses[-1] > 0.01:
                        has_improved = True
                if not has_improved:
                    epoch = 10000000  # End
            if verbose:
                epoch_trloss = np.mean([i[1] for i in self.log['training_loss'] if i[0] == epoch])
                epoch_tracc = np.mean([i[1].cpu() for i in self.log['training_accuracy'] if i[0] == epoch])
                print(f"{epoch}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}")
            epoch += 1
        self.eval()
        self.to('cpu')

    def predict(self, x, device='cpu'):
        self.to(device)
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward(x.to(device))
        return np.array([p.argmax().item() for p in preds])

    def predict_proba(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward(x.to('cpu'))
        return preds.detach().cpu().numpy()
