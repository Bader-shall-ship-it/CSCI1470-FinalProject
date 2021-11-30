import torch
import torchvision
import os
import sys
import argparse
import numpy as np

from models.simclr import SimCLRModel
from data.dataloaders import CIFAR10_dataloader, ImageNet_dataloader
from models.losses import NTXent
from train import train


# Hyperparameters
# TODO: Find somewhere else for these; also, fix batch size, this is placeholder
BATCH_SIZE = 32
EPOCHS = 100


def parse_args() -> argparse.Namespace:
  """Parse arguments from command line into ARGS."""

  parser = argparse.ArgumentParser(
      description="The runner for our SimCLR implementation",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
      '--data',
      default='cifar',
      help='Dataset to use for training and testing',
      choices=['cifar', 'imagenet'],
      dest='data'
  )

  parser.add_argument(
      '--no-augment',
      default=False,
      help='Do not perform data augmentations',
      action='store_true',
      dest='noaug'
  )

  return parser.parse_args()


def main() -> None:
  # Get device
  active_device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {active_device}")

  # Load data
  if (args.data == "cifar"):
    print("Loading cifar dataset")
    train_loader, test_loader = CIFAR10_dataloader(~args.noaug, BATCH_SIZE)
    using_cifar = True
  elif (args.data == "imagenet"):
    print("Loading imagenet dataset")
    train_loader, test_loader = ImageNet_dataloader(~args.noaug, BATCH_SIZE)
    using_cifar = False
  else:
    sys.exit("Should not have gotten here.")

  # Instantiate model
  model = SimCLRModel(CIFAR=using_cifar, device=active_device)

  # Train
  for epoch in range(EPOCHS):
    print("Now on epoch " + str(epoch) + "/" + str(EPOCHS))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = NTXent(BATCH_SIZE, active_device, tau=1)

    train(model, train_loader, optimizer, active_device, loss_fn)


if __name__ == "__main__":
  args = parse_args()
  main()
