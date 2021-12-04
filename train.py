import torch
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


def test_classifier(model: nn.Module, data_loader: DataLoader, device: str) -> None:
    """Test a classifier."""
    num_batches = len(data_loader)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_loss += F.cross_entropy(logits, y)
            correct += (logits.argmax(dim=1) ==
                        y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= len(data_loader.dataset)
    print(
        f"Test error: \n Accuracy: {(100*correct):>0.1f}%, average loss: {test_loss:>8f} \n")