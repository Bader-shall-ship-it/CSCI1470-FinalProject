import argparse
import os
import sys

import numpy as np
import torch
import torchvision

from data.dataloaders import CIFAR10_dataloader, ImageNet_dataloader
from models.losses import NTXent
from models.simclr import SimCLRModel
from train import train


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our SimCLR implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--action',
        required=True,
        help='Whether to train or to test',
        choices=['train', 'test'],
        dest='action'
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
        '--batch-size',
        default=32,
        type=int,
        help='Batch size used for training',
        dest='batch_size'
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


def train() -> None:
    """Helper function to store logic for train action."""
    pass


def test() -> None:
    """Helper function to store logic for test action."""
    pass


def main(args: argparse.Namespace) -> None:
    # Get device
    active_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {active_device}")

    # Load data
    if (args.data == "cifar"):
        print("Loading cifar dataset")
        train_loader, test_loader = CIFAR10_dataloader(~args.noaug, args.batch_size)
        using_cifar = True
    elif (args.data == "imagenet"):
        print("Loading imagenet dataset")
        train_loader, test_loader = ImageNet_dataloader(
            ~args.noaug, args.batch_size)
        using_cifar = False

    # Instantiate model
    model = SimCLRModel(CIFAR=using_cifar, device=active_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = NTXent(args.batch_size, active_device, tau=1)

    checkpoint_path = "./checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    # Train
    for epoch in range(1, args.epochs + 1):
        print("Now on epoch " + str(epoch) + "/" + str(args.epochs))
        train(model, train_loader, optimizer, active_device, loss_fn)

        torch.save(model.state_dict(), checkpoint_path +
                   "model_{}.pth".format(epoch))


if __name__ == "__main__":
    args = parse_args()
    main(args)
