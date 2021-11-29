import torch
import torchvision
import os
import sys
import random
from torch import nn
from tqdm import tqdm

# Types
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# Hyperparameters
# TODO: find somewhere else for these
EPOCHS = 100


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer) -> None:
  for epoch in range(EPOCHS):
    for x, y in tqdm(data_loader):
      pass


def main() -> None:
  pass


if __name__ == "__main__":
  main()
