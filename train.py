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


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, device: str) -> None:
    """Train model for a single epoch"""
    for x, _ in tqdm(data_loader, total=len(data_loader.dataset)):
        e_i = x[0]
        e_j = x[1]

    return


def main() -> None:
    pass


if __name__ == "__main__":
    main()
