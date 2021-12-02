import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
# Types
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, device: str, loss_fn: nn.Module) -> None:
    """Train contrastive model for a single epoch"""
    model.train()

    for batch, (x, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Send tensors to GPU
        x_i = x[0]
        x_j = x[1]
        assert x_i.shape == x_j.shape

        x_i, x_j = x_i.to(device), x_j.to(device)

        # Compute loss
        z_i = model(x_i)
        z_j = model(x_j)
        loss = loss_fn(z_i, z_j)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Loss: {loss.item():>7f}")

    return


def train_classifier(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, device: str) -> None:
    '''Train classification model for a single epoch'''

    model.train()
    for i, (x, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Loss: {loss.item():>7f}")
