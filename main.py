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
      '--epochs',
      default=10,
      type=int,
      help='Number of training epochs',
  )

  parser.add_argument(
      '--batch_size',
      default=32,
      type=int,
      help='Batch size used for training',
  )

  parser.add_argument(
      '--lr',
      default=1e-3,
      type=float,
      help='Optimizer learning rate',
  )

  parser.add_argument(
      '--no-augment',
      default=False,
      help='Do not perform data augmentations',
      action='store_true',
      dest='noaug'
  )

  return parser.parse_args()


def main(args: argparse.Namespace) -> None:
  # Get device
  active_device = "cuda" if torch.cuda.is_available() else "cpu"
  batch_size = args.batch_size
  epochs = args.epochs
  print(f"Using device: {active_device}")

  # Load data
  if (args.data == "cifar"):
    print("Loading cifar dataset")
    train_loader, test_loader = CIFAR10_dataloader(~args.noaug, batch_size)
    using_cifar = True
  elif (args.data == "imagenet"):
    print("Loading imagenet dataset")
    train_loader, test_loader = ImageNet_dataloader(~args.noaug, batch_size)
    using_cifar = False

  # Instantiate model
  model = SimCLRModel(CIFAR=using_cifar, device=active_device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  loss_fn = NTXent(batch_size, active_device, tau=1)
  
  # Train
  for epoch in range(epochs):
    print("Now on epoch " + str(epoch) + "/" + str(epochs))
    train(model, train_loader, optimizer, active_device, loss_fn)


if __name__ == "__main__":
  args = parse_args()
  main(args)
