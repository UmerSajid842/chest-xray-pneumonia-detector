"""Model construction for the chest X-ray classifier."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Build a ResNet-18 classifier with an explicit pretrained toggle.

    Set pretrained=True only when the environment can download the public
    torchvision weights and the use of those weights is acceptable.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
