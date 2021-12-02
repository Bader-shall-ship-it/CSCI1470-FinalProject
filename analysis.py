import torch
from torch import nn
from models.simclr import SimCLRModel, SimCLRClassifier
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
from train import train_classifier

# TODO(Bader): General flow of the program. Not tested yet.
def benchmark_simclr_backbone(small_train_loader: DataLoader, small_test_loader: DataLoader, num_classes: int, simclr_weights_path: str, CIFAR: bool, device: torch.device):

    # Load weights and extract backbone.
    # We could also just save backbone weights directly and instantiate ResnetBackbone
    simclr = SimCLRModel(CIFAR=CIFAR)
    simclr.load_state_dict(torch.load(simclr_weights_path))

    simclr_classifier = SimCLRClassifier(
        num_classes=num_classes, CIFAR=CIFAR, device=device)
    simclr_classifier.feature = simclr.feature

    optimizer = torch.optim.Adam(simclr_classifier.parameters(), lr=0.001)

    train_classifier(simclr_classifier, data_loader=small_train_loader,
                     optimizer=optimizer, device=device)

    # Get accuracy on heldout set, and hope its high.
    test_classifier(simclr_classifier, data_loader=small_test_loader,
                    optimizer=optimizer, device=device)


def test_classifier(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, device: str) -> None:
    """Test a classifier."""
    size = len(data_loader.datset)
    num_batches = len(data_loader)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_loss += F.cross_entropy(logits, y)
            correct += (logits.argmax(dim=1) ==
                        y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= num_batches
    print(
        f"Test error: \n Accuracy: {(100*correct):>0.1f}%, average loss: {test_loss:>8f} \n")
