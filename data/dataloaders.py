from typing import Tuple

import PIL
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader

from data.utils import generate_augmented_pair, transform_test_img

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR10_MEAN = [0.491, 0.482, 0.446]
CIFAR10_STD = [0.247, 0.2434, 0.2615]


def CIFAR10_dataloader(augment: bool, batch_size: int, root: str = './dataset') -> Tuple[DataLoader, DataLoader]:
    '''Create train and test dataloaders for CIFAR10'''

    train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=augment_CIFAR10 if augment else transform_CIFAR10_test)
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_CIFAR10_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def ImageNet_dataloader(augment: bool, batch_size: int, root: str = './dataset') -> Tuple[DataLoader, DataLoader]:
    '''Create train and test dataloaders for ImageNet'''
    # TODO(Bader): ImageNet is not longer publically accesible. Need to manually download the dataset and place in root.
    train_dataset = torchvision.datasets.ImageNet(
        root=root, train=True, download=True, transform=augment_ImageNet if augment else transform_ImageNet_test)
    test_dataset = torchvision.datasets.ImageNet(
        root=root, train=False, download=True, transform=transform_ImageNet_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def augment_CIFAR10(img: PIL.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    return generate_augmented_pair(img, 32, CIFAR10_MEAN, CIFAR10_STD)


def augment_ImageNet(img: PIL.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    return generate_augmented_pair(img, 224, IMAGENET_MEAN, IMAGENET_STD)


def transform_CIFAR10_test(img: PIL.Image) -> torch.Tensor:
    return transform_test_img(img, CIFAR10_MEAN, CIFAR10_STD)


def transform_ImageNet_test(img: PIL.Image) -> torch.Tensor:
    return transform_test_img(img, IMAGENET_MEAN, IMAGENET_STD)
