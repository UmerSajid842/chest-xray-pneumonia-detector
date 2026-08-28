import torch

from src.inference import WARNING
from src.model import build_model


def test_model_has_two_class_output():
    model = build_model(num_classes=2, pretrained=False)
    output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 2)


def test_inference_warning_is_non_diagnostic():
    assert "not a medical diagnosis" in WARNING
