"""Inference utilities for a trained chest X-ray classifier."""
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .model import build_model

WARNING = "This output is an educational model estimate, not a medical diagnosis."


def predict_image(image_path: Path, checkpoint_path: Path, threshold: float = 0.5) -> dict:
    """Predict NORMAL or PNEUMONIA from one image using a saved checkpoint."""
    _, eval_tfms = build_transforms()
    model = build_model(num_classes=2, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        probabilities = torch.softmax(model(eval_tfms(image).unsqueeze(0)), dim=1)[0]
    pneumonia_probability = float(probabilities[1])
    return {
        "predicted_class": "PNEUMONIA" if pneumonia_probability >= threshold else "NORMAL",
        "probabilities": {"NORMAL": float(probabilities[0]), "PNEUMONIA": pneumonia_probability},
        "threshold": threshold,
        "warning": WARNING,
    }
