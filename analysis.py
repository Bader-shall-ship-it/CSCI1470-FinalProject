import torch
from models.simclr import SimCLRModel, SimCLRClassifier
from torch.utils.data import DataLoader
from train import train_classifier

# TODO(Bader): General flow of the program. Not tested yet.
def benchmark_simclr_backbone(small_train_loader: DataLoader, small_test_loader: DataLoader, num_classes: int, simclr_weights_path: str, CIFAR: bool, device: torch.device):

    # Load weights and extract backbone.
    # We could also just save backbone weights directly and instantiate ResnetBackbone
    simclr = SimCLRModel(CIFAR=CIFAR)
    simclr.load_state_dict(torch.load(simclr_weights_path))

    simclr_classifier = SimCLRClassifier(num_classes=num_classes, CIFAR=CIFAR, device=device)
    simclr_classifier.feature = simclr.feature

    optimizer = torch.optim.Adam(simclr_classifier.parameters(), lr=0.001)

    train_classifier(simclr_classifier, data_loader=small_train_loader, optimizer=optimizer, device=device)

    # Get accuracy on heldout set, and hope its high.