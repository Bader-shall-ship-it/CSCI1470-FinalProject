import torch
import torchvision.transforms as transforms
import PIL
from typing import Tuple, List
 
def generate_augmented_pair(img: PIL.Image, crop_resize: int, normalize_mean: List[float], normalize_std: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    s = 1.0
    tr = transforms.Compose([
        transforms.RandomResizedCrop(crop_resize),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
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