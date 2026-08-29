"""Small reproducible training run for the Kaggle chest X-ray dataset."""
from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data import build_dataloaders
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0, help="Bound batches per split for a quick smoke run; 0 means all batches.")
    parser.add_argument("--eval-all", action="store_true", help="Use all validation and test batches while keeping bounded training.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(args.data_root, batch_size=args.batch_size, num_workers=0)
    model = build_model(num_classes=2, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, total = 0.0, 0
        for batch_index, (images, labels) in enumerate(loaders["train"]):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loaders["val"]):
                labels = labels.to(device)
                loss = criterion(model(images.to(device)), labels)
                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
                if args.max_batches and not args.eval_all and batch_index + 1 >= args.max_batches:
                    break
        record = {"epoch": epoch + 1, "train_loss": total_loss / total, "val_loss": val_loss / max(val_total, 1)}
        history.append(record)
        print(record)
    checkpoint = args.output_dir / "pneumonia_detector.pt"
    torch.save(model.state_dict(), checkpoint)
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loaders["test"]):
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu()
            y_true.extend(labels.tolist())
            y_score.extend(probs.tolist())
            y_pred.extend((probs >= 0.5).int().tolist())
            if args.max_batches and not args.eval_all and batch_index + 1 >= args.max_batches:
                break
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["NORMAL", "PNEUMONIA"], output_dict=True, zero_division=0)
    metrics = {"device": str(device), "dataset_root": str(args.data_root), "class_to_idx": loaders["train"].dataset.class_to_idx, "history": history, "classification_report": report, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(), "checkpoint": str(checkpoint)}
    if len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
