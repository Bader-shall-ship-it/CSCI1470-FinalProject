import torch
import torchvision

class ResnetBackbone(torch.nn.Module):
    def __init__(self, pretrained: bool, CIFAR: bool):
        super(ResnetBackbone, self).__init__()

        resnet = list(torchvision.models.resnet50(pretrained=pretrained).children())
        # TODO: this is brittle.
        if CIFAR:
            # Change first conv from 7x7 conv of stride 2 with 3x3 conv of stride 1
            resnet[0] = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1,1), padding=1, bias=False)
            # Remove first maxpool
            resnet.pop(3)

        # Remove classification head
        resnet = resnet[:-1]
        self.model = torch.nn.Sequential(*resnet)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
        