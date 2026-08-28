# Model Card: Chest X-Ray Pneumonia Detector

## Intended use

Educational demonstration of CNN-based image classification engineering. The model is not intended for diagnosis, triage, treatment decisions, or deployment in clinical workflows.

## Inputs and outputs

The model accepts a chest X-ray image after grayscale-to-RGB conversion, resizing, tensor conversion, and normalization. It returns class probabilities for `NORMAL` and `PNEUMONIA`, a configurable threshold, and a non-diagnostic warning.

## Limitations

Performance depends on the dataset, split policy, scanner population, label quality, and preprocessing. Shortcut learning, demographic imbalance, site shift, and calibration error may materially affect results. No fixed benchmark is claimed until a reproducible dataset run and evaluation report are added.

## Ethical and privacy considerations

Do not commit patient images or personal information. Use only data with appropriate permissions and licensing. Any clinical or public-health use requires independent validation, governance, regulatory review, and qualified medical oversight.
