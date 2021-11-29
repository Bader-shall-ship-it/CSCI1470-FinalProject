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


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer) -> None:
  """Train model for a single epoch"""
  for batch_num, (x, y) in tqdm(enumerate(data_loader)):
    print("We are currently on batch " + str(batch_num))

  return


def main() -> None:
  pass


if __name__ == "__main__":
  main()
