import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

class FFNN(nn.Module):
    def __init__(self, input_dim, n_classes=5, topology=[256,128], dropout=0.3):
        super().__init__()
        layers = []
        dims   = [input_dim] + topology

        for i in range(len(topology)):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate_loss_accuracy(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb  = xb.to(device), yb.to(device)
            logits  = model(xb)
            loss    = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return total_loss / total, correct / total


class FFNNv2(nn.Module):
    """FFNN melhorada com BatchNorm, skip-connections e activação configurável."""

    def __init__(self, input_dim, n_classes=5, topology=[512, 256, 128],
                 dropout=0.3, activation='gelu'):
        super().__init__()
        act_fn = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'leakyrelu': nn.LeakyReLU}
        Act = act_fn.get(activation, nn.GELU)

        self.input_proj = nn.Linear(input_dim, topology[0])
        self.input_bn = nn.BatchNorm1d(topology[0])
        self.input_act = Act()
        self.input_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for i in range(len(topology) - 1):
            block = nn.ModuleDict({
                'linear': nn.Linear(topology[i], topology[i + 1]),
                'bn': nn.BatchNorm1d(topology[i + 1]),
                'act': Act(),
                'drop': nn.Dropout(dropout),
            })
            if topology[i] != topology[i + 1]:
                block['skip'] = nn.Linear(topology[i], topology[i + 1], bias=False)
            self.blocks.append(block)

        self.head = nn.Linear(topology[-1], n_classes)

    def forward(self, x):
        x = self.input_drop(self.input_act(self.input_bn(self.input_proj(x))))
        for block in self.blocks:
            identity = x
            out = block['drop'](block['act'](block['bn'](block['linear'](x))))
            if 'skip' in block:
                identity = block['skip'](identity)
            x = out + identity
        return self.head(x)


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, class_weight=None, weight_decay=1e-4):
    weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device) if class_weight is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc, patience_counter = 0, 0
    best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate_loss_accuracy(model, val_loader, criterion)

        if epoch % 10 == 0:
            print(f'[Epoch {epoch:03d}] train_acc: {tr_acc:.4f} | val_acc: {vl_acc:.4f}')

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(best_state)
    return model