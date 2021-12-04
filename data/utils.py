from typing import List, Tuple

import PIL
import torch
import torchvision.transforms as transforms
import PIL
from typing import Tuple, List
from torch.utils.data import DataLoader

def generate_augmented_pair(img: PIL.Image, crop_resize: int, normalize_mean: List[float], normalize_std: List[float], strength: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    tr = transforms.Compose([
        transforms.RandomResizedCrop(crop_resize),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(
            0.8*strength, 0.8*strength, 0.8*strength, 0.2*strength)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=0.1*img.height),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)])

    return tr(img), tr(img)


def transform_test_img(img: PIL.Image, normalize_mean: List[float], normalize_std: List[float]) -> torch.Tensor:
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)])
    
    return tr(img)

def create_dataloader_subset(dataloader: DataLoader, train_size: int, test_size: int, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    train_dataset = torch.utils.data.Subset(dataloader.dataset, torch.arange(train_size))
    test_dataset = torch.utils.data.Subset(dataloader.dataset, torch.arange(train_size, test_size))

    new_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    new_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    return new_train, new_test
