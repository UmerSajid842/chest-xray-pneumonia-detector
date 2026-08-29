# Chest X-Ray Pneumonia Detector

A CNN-based medical computer-vision project for binary pneumonia classification from chest X-ray images. The project was listed on Umer Sajid’s CV as a model deployed through a public Hugging Face Gradio application. This repository organizes the work into a reviewable, reproducible structure with a notebook, reusable inference code, tests, and deployment guidance.

> **Important medical-use boundary:** This is an educational machine-learning project, not a clinical diagnostic device. A model output must never be used as the sole basis for patient care. Any real-world use would require clinical validation, representative data, calibration, fairness analysis, regulatory review, and qualified medical oversight.

## Project overview

The task is binary image classification: given a chest X-ray image, estimate whether the image belongs to the pneumonia or normal class. The implementation uses transfer learning with a convolutional neural network, explicit train/validation/test folders, image normalization, and a probability threshold that can be reviewed rather than hidden inside the UI.

The notebook explains the end-to-end workflow: dataset contract, preprocessing, model construction, training loop design, evaluation metrics, error analysis, and export of a model checkpoint. The source package keeps model creation and preprocessing separate from any user interface.

## Repository structure

```text
.
├── src/
│   ├── __init__.py
│   ├── data.py          # Dataset layout, transforms, and dataloaders
│   ├── model.py         # Transfer-learning CNN factory
│   └── inference.py     # Single-image prediction helper
├── notebooks/
│   └── 01_training_and_evaluation.ipynb
├── tests/
│   └── test_model_and_inference.py
├── docs/
│   ├── assets/
│   │   ├── sample-normal.jpeg
│   │   └── sample-pneumonia.jpeg
│   ├── MODEL_CARD.md
│   └── SMOKE_RUN.md
├── scripts_train.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset contract

The code expects an **untracked** dataset directory with this layout. The dataset used for the first smoke run was downloaded exactly with the KaggleHub command below:

```python
import kagglehub

path = kagglehub.dataset_download("ghost5612/chest-x-ray-images-normal-and-pneumonia")
print("Path to dataset files:", path)
```

The source is the [Chest X-Ray Images (Normal and Pneumonia) Kaggle dataset](https://www.kaggle.com/datasets/ghost5612/chest-x-ray-images-normal-and-pneumonia). The full dataset is intentionally not committed. Review the current Kaggle terms before redistribution or deployment.

Observed image counts from the downloaded version were: train `NORMAL=1,341`, `PNEUMONIA=3,875`; validation `NORMAL=24`, `PNEUMONIA=23`; test `NORMAL=234`, `PNEUMONIA=390`; total `5,887` JPEG images.

The code expects an **untracked** dataset directory with this layout:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Do not commit patient images or personally identifiable information. Confirm that the dataset license permits the intended educational use. The repository intentionally contains no medical images and no fixed performance claim.

## Local setup

```bash
git clone https://github.com/UmerSajid842/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebook with Jupyter:

```bash
jupyter notebook notebooks/01_training_and_evaluation.ipynb
```

The notebook is designed to stop clearly when the dataset is not present rather than silently using fabricated data. Set `DATA_ROOT` in the first configuration cell to your local dataset directory.

## Inference example

After training and saving a compatible checkpoint, the reusable inference helper can be used as follows:

```python
from pathlib import Path
from src.inference import predict_image

result = predict_image(
    image_path=Path("sample_xray.jpeg"),
    checkpoint_path=Path("artifacts/pneumonia_detector.pt"),
)
print(result)
```

The returned object includes the predicted class, class probabilities, threshold, and a warning that the result is not a medical diagnosis.

## Dataset examples

The following representative images are included only to make the README and notebook easier to understand. They are source dataset examples, not model predictions or medical guidance.

| NORMAL class example | PNEUMONIA class example |
| --- | --- |
| ![Representative normal chest X-ray](docs/assets/sample-normal.jpeg) | ![Representative pneumonia-class chest X-ray](docs/assets/sample-pneumonia.jpeg) |

## Dataset-backed smoke run

A one-training-batch CPU smoke run was completed against the downloaded dataset, followed by full validation and test evaluation. It produced `train_loss=0.5903142094612122`, `val_loss=0.8002765026498349`, test accuracy `0.375`, and ROC-AUC `0.7438965592811746`. The confusion matrix was `[[234, 0], [390, 0]]`; the model predicted every test image as `NORMAL`.

These results are **not a benchmark** because only one training batch was used. They are included to show a real, reproducible execution and the importance of balanced training and error analysis. Full command and output are documented in [docs/SMOKE_RUN.md](docs/SMOKE_RUN.md).

## Evaluation expectations

For a trustworthy experiment, report sensitivity/recall, specificity, precision, F1, ROC-AUC, confusion matrix, calibration, and subgroup or site-level error analysis where metadata and permissions allow. Accuracy alone is not sufficient for a medical-imaging classifier. This repository does not assert a benchmark until a reproducible run with a documented dataset split and saved evaluation report is added.

## Hugging Face deployment note

The CV describes a public Hugging Face Gradio deployment. The repository now includes the dataset-backed training path and sample documentation images, but no unverified live deployment URL is claimed here. A deployment can be recreated by wrapping `predict_image` in a Gradio interface and loading a user-provided checkpoint through a secret or a private artifact store. Do not upload patient data, credentials, or an unreviewed clinical model. The notebook and source code here are the canonical implementation scaffold; deployment URLs should be added only after they are verified.

## Limitations

Chest X-ray datasets can contain shortcut signals, institution-specific artifacts, demographic imbalance, label noise, and distribution shift. A model that performs well on one test split may fail on another hospital, scanner, age group, or disease presentation. This project should be presented as a portfolio demonstration of computer-vision engineering, not as a validated medical product.

## License

MIT License. See [LICENSE](LICENSE). Dataset terms remain separate and must be followed independently.
