import torch
import torch.nn as nn

from .modules import ResnetBackbone


class SimCLRModel(nn.Module):
    ''' SimCLR model for contrastive learning'''
    
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 1024, pretrain_backbone: bool = False, CIFAR: bool = True, device: torch.device = "cpu"):
        super().__init__()
        # TODO: Double check the hidden_dim in the paper.
        self.feature = ResnetBackbone(pretrained=pretrain_backbone, CIFAR=CIFAR).to(device)
        self.g = nn.Sequential(nn.Linear(2048, hidden_dim, bias=False), 
                              nn.BatchNorm1d(hidden_dim),
                              nn.ReLU(), 
                              nn.Linear(hidden_dim, feature_dim)).to(device)
  
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.feature(input)
        x = x.flatten(start_dim=1, end_dim=-1)
        output = self.g(x)
        return output


class SimCLRClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrain_backbone: bool = False, CIFAR: bool = True, device: torch.device = "cpu"):
        super().__init__()
        self.feature = ResnetBackbone(pretrained=pretrain_backbone, CIFAR=CIFAR).to(device)
        self.fc = nn.Linear(2048, num_classes, bias=True).to(device)
  
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.feature(input)
        x = x.flatten(start_dim=1, end_dim=-1)
        output = self.fc(x)
        return output
