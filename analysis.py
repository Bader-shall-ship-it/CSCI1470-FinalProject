import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.simclr import SimCLRClassifier, SimCLRModel
from train import train_classifier, test_classifier

def benchmark_simclr_backbone(small_train_loader: DataLoader, small_test_loader: DataLoader, num_classes: int, epochs: int, simclr_weights_path: str, CIFAR: bool, device: torch.device):

    # Load weights and extract backbone.
    simclr = SimCLRModel(CIFAR=CIFAR)
    simclr.load_state_dict(torch.load(simclr_weights_path))

    simclr_classifier = SimCLRClassifier(
        num_classes=num_classes, CIFAR=CIFAR, device=device)
    simclr_classifier.feature = simclr.feature.to(device)

    optimizer = torch.optim.Adam(simclr_classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_classifier(simclr_classifier, data_loader=small_train_loader,
                        optimizer=optimizer, device=device)

    # Get accuracy on heldout set, and hope its high.
    test_classifier(simclr_classifier, data_loader=small_test_loader,
                    device=device)

