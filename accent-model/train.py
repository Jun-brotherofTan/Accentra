"""
train.py — Training script for the Accentra accent classifier.

Usage
-----
Normal training (requires preprocessed dataset manifest):
    python train.py \
        --manifest accent-model/data/processed/manifest.json \
        --use-wav2vec \
        --epochs 20 \
        --batch-size 16 \
        --output accent-model/model.pt

Demo / CI mode (generates synthetic data — no real dataset required):
    DEMO_MODE=1 python train.py --epochs 5 --output accent-model/model.pt

The trained model is saved as a checkpoint dict with keys:
    model_state_dict, optimizer_state_dict, epoch, val_accuracy, config
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import AccentClassifier, ACCENT_CLASSES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class AccentDataset(Dataset):
    """
    Loads pre-computed feature vectors (.npy) from a manifest.json file.

    manifest.json format (produced by preprocess.py):
    [
      {"feature_file": "...", "label_idx": 0, "accent": "korean", ...},
      ...
    ]
    """

    def __init__(self, manifest_path: str):
        with open(manifest_path) as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        features = np.load(record["feature_file"]).astype(np.float32)
        label = int(record["label_idx"])
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)


class SyntheticAccentDataset(Dataset):
    """
    Synthetic dataset for demo / CI purposes.

    Generates random MFCC-like feature vectors with a slight per-class offset
    so the model can learn something meaningful in demo mode.
    NOTE: Model trained on synthetic data will NOT predict real accents.
    """

    def __init__(self, num_samples: int = 400, mfcc_dim: int = 40):
        self.samples: List[Tuple[np.ndarray, int]] = []
        rng = np.random.default_rng(seed=42)
        per_class = num_samples // NUM_CLASSES
        for class_idx in range(NUM_CLASSES):
            # Each class has a distinct mean to make classification feasible
            mean = np.zeros(mfcc_dim)
            mean[class_idx * (mfcc_dim // NUM_CLASSES)] = float(class_idx + 1)
            for _ in range(per_class):
                feat = rng.normal(loc=mean, scale=0.5).astype(np.float32)
                self.samples.append((feat, class_idx))
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat, label = self.samples[idx]
        return torch.tensor(feat), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the Accentra accent classifier")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.json produced by preprocess.py. "
             "If omitted (or DEMO_MODE=1), synthetic data is used.",
    )
    parser.add_argument(
        "--use-wav2vec",
        action="store_true",
        help="Use wav2vec2 encoder (requires transformers + internet on first run). "
             "When False, expects MFCC features in the manifest.",
    )
    parser.add_argument("--mfcc-dim", type=int, default=40, help="MFCC feature dim (default 40)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument(
        "--output",
        default="model.pt",
        help="Where to save the trained model checkpoint",
    )
    args = parser.parse_args()

    demo_mode = bool(os.environ.get("DEMO_MODE")) or args.manifest is None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------ data
    if demo_mode:
        print(
            "[DEMO MODE] Training on SYNTHETIC data. "
            "This model will NOT predict real accents accurately."
        )
        dataset = SyntheticAccentDataset(num_samples=400, mfcc_dim=args.mfcc_dim)
        use_wav2vec = False
    else:
        print(f"Loading dataset from {args.manifest}")
        dataset = AccentDataset(args.manifest)
        use_wav2vec = args.use_wav2vec

    # Train/val split
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # ----------------------------------------------------------------- model
    model = AccentClassifier(
        num_classes=NUM_CLASSES,
        freeze_encoder=True,
        use_wav2vec=use_wav2vec,
        mfcc_dim=args.mfcc_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # --------------------------------------------------------------- training
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss {train_loss:.4f}  acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f}  acc {val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_accuracy": val_acc,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "accent_classes": ACCENT_CLASSES,
                    "use_wav2vec": use_wav2vec,
                    "mfcc_dim": args.mfcc_dim,
                    "demo_mode": demo_mode,
                },
            }

    # ------------------------------------------------------------------ save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(output_path))
    print(f"\nModel saved to {output_path}  (best val acc: {best_val_acc:.3f})")


if __name__ == "__main__":
    main()
