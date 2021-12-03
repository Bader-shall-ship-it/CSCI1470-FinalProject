import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from analysis import benchmark_simclr_backbone
from data.dataloaders import CIFAR10_dataloader, ImageNet_dataloader
from data.utils import create_dataloader_subset
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
        '--train',
        default=True,
        help='Train',
        action='store_true',
        dest='train'
    )

    parser.add_argument(
        '--test',
        help='Test',
        action='store_false',
        dest='train'
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

    parser.add_argument(
        '--weights',
        default='',
        help='Path for the weights to use in testing',
        dest='weights'
    )

    return parser.parse_args()


def load_data(train: bool, batch_size: int, augment: bool) -> Tuple[DataLoader, DataLoader, bool]:
    """Loads data for either training or testing."""
    # Create dataloaders
    if (args.data == "cifar"):
        print("Loading cifar dataset")
        train_loader, test_loader = CIFAR10_dataloader(augment, batch_size)
        using_cifar = True
        num_classes = 10
    elif (args.data == "imagenet"):
        print("Loading imagenet dataset")
        train_loader, test_loader = ImageNet_dataloader(augment, batch_size)
        using_cifar = False
        # TODO: no idea how many classes imagenet 2012 has but idt we using it so its should be ok
        num_classes = 1000

    # Reduce dataset if testing
    if not train:
        size = int(len(train_loader.dataset))
        train_loader, test_loader = create_dataloader_subset(test_loader, size // 10, size // 10, batch_size=args.batch_size)

    return train_loader, test_loader, using_cifar, num_classes


def main(args: argparse.Namespace) -> None:
    # Get device
    active_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {active_device}")

    # Load data
    train_loader, test_loader, using_cifar, num_classes = load_data(args.train)

    if (args.train):
        # Train
        model = SimCLRModel(CIFAR=using_cifar, device=active_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = NTXent(args.batch_size, active_device, tau=1)

        checkpoint_path = "./checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        # Train
        for epoch in range(args.epochs):
            print("Now on epoch " + str(epoch) + "/" + str(args.epochs))
            train(model, train_loader, optimizer, active_device, loss_fn)

            torch.save(model.state_dict(), checkpoint_path +
                       "model_{}.pth".format(epoch))
    else:
        # Test
        if (args.weights == ''):
            print("Weights not specified for testing")
        else:
            benchmark_simclr_backbone(train_loader, test_loader, num_classes=num_classes,
                                      simclr_weights_path=args.weights, CIFAR=using_cifar, device=active_device)


if __name__ == "__main__":
    args = parse_args()
    print(args.train)
    main(args)
