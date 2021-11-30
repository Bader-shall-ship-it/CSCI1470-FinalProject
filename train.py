import torch
import torchvision
import os
import sys
import random
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt

# Types
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# TODO: put this somewhere better
BATCH_SIZE = 32

def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, device: str, loss_fn: nn.Module) -> None:
    """Train model for a single epoch"""
    size = len(data_loader.dataset)

    model.train()
    for batch, (x, _) in tqdm(enumerate(data_loader), total=size // BATCH_SIZE):
        # Send tensors to GPU
        x_i = x[0]
        x_j = x[1]

        if (x_i.shape != x_j.shape):
            print("Houston, we have a problem")
            print("Shape of x_i: " + x_i.shape)
            print("Shape of x_j: " + x_j.shape)
            sys.exit()

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
            loss, current = loss.item(), batch * len(x)
            print(f"Loss: {loss:>7f}")

    return


def main() -> None:
    pass


if __name__ == "__main__":
    main()
