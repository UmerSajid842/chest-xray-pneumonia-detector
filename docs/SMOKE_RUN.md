# Dataset-backed smoke run

## Dataset source

The dataset was downloaded with KaggleHub using:

```python
import kagglehub
path = kagglehub.dataset_download("ghost5612/chest-x-ray-images-normal-and-pneumonia")
```

Source page: [Chest X-Ray Images (Normal and Pneumonia) on Kaggle](https://www.kaggle.com/datasets/ghost5612/chest-x-ray-images-normal-and-pneumonia). The full dataset remains outside this repository. Review the current Kaggle terms before redistribution or deployment.

## Observed dataset layout

| Split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| train | 1,341 | 3,875 | 5,216 |
| val | 24 | 23 | 47 |
| test | 234 | 390 | 624 |
| **Total** | **1,599** | **4,288** | **5,887** |

## Reproduction command

The following command performed one training batch and evaluated the complete validation and test splits on CPU:

```bash
python3 scripts_train.py \
  --data-root /path/to/chest_xray \
  --output-dir artifacts \
  --epochs 1 \
  --batch-size 32 \
  --max-batches 1 \
  --eval-all
```

## Observed output

The run produced `train_loss=0.5903142094612122` and `val_loss=0.8002765026498349` after one training batch. On the 624-image test split, the run produced accuracy `0.375` and ROC-AUC `0.7438965592811746`; the confusion matrix was `[[234, 0], [390, 0]]`.

These numbers are **not a model benchmark**. The model saw only one training batch, predicted every test image as `NORMAL`, and therefore demonstrates why a longer, balanced, reproducible training protocol and threshold analysis are required. Do not use these outputs for medical decisions.
