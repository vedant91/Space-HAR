"""
LSTM Sequence Classifier — Training Script
==========================================
Trains an LSTM model on skeleton sequences to classify experiment steps.

Usage:
    python train/train_lstm.py

Colab-ready: Upload dataset/skeleton_sequences/ and run.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Load config ───────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    TRAIN_VAL_SPLIT, SEQUENCE_WINDOW,
    NUM_STEPS, LSTM_PATH, LSTM_ONNX_PATH, EXPERIMENT_STEPS,
)

# Feature dimension: MediaPose only (33×4 = 132)
# Pose only (no hands) for faster inference
FEATURE_DIM = 33 * 4  # 132


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class SkeletonSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, window_size, feature_dim)
        y: (N,) — step IDs (1-indexed)
        """
        # Map step IDs to 0-indexed classes
        unique_labels = sorted(np.unique(y))
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self.idx_to_label = {idx: lbl for lbl, idx in self.label_to_idx.items()}

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor([self.label_to_idx[int(label)] for label in y])
        self.num_classes = len(unique_labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class HARLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for HAR step classification.
    Input: (batch, seq_len, feature_dim)
    Output: (batch, num_classes)
    """
    def __init__(self, feature_dim: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input normalization
        self.input_bn = nn.BatchNorm1d(feature_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attention = nn.Linear(hidden_size * 2, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch, seq, feat = x.shape

        # Normalize features
        x_reshaped = x.reshape(batch * seq, feat)
        x_normed = self.input_bn(x_reshaped).reshape(batch, seq, feat)

        # LSTM
        lstm_out, _ = self.lstm(x_normed)  # (batch, seq, hidden*2)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        # Classify
        return self.classifier(context)


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_model(data_dir: str = "dataset/skeleton_sequences",
                output_path: str = None,
                epochs: int = None,
                hidden_size: int = None,
                dropout: float = None,
                seed: int = 7):
    if output_path is None:
        output_path = LSTM_PATH
    epochs = LSTM_EPOCHS if epochs is None else epochs
    hidden_size = LSTM_HIDDEN_SIZE if hidden_size is None else hidden_size
    dropout = LSTM_DROPOUT if dropout is None else dropout
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load data ─────────────────────────────────────────────
    X_path = Path(data_dir) / "X_sequences.npy"
    y_path = Path(data_dir) / "y_labels.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_dir}.\n"
            "Run mediapipe_labeler.py first to generate sequences."
        )

    X = np.load(str(X_path))
    y = np.load(str(y_path))
    logger.info("Loaded dataset: X=%s, y=%s", X.shape, y.shape)
    logger.info("Class distribution: %s", {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})

    # ── Dataset split ─────────────────────────────────────────
    dataset = SkeletonSequenceDataset(X, y)
    num_classes = dataset.num_classes
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False)
    logger.info("Train: %d | Val: %d | Classes: %d", train_size, val_size, num_classes)

    # ── Model ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    model = HARLSTMClassifier(
        feature_dim=X.shape[-1],
        hidden_size=hidden_size,
        num_layers=LSTM_NUM_LAYERS,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────
    best_val_acc = -1.0
    patience_counter = 0
    max_patience = 15

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %03d/%03d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.3f",
                epoch, epochs, train_loss / len(train_loader),
                val_loss / len(val_loader), val_acc
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "feature_dim": X.shape[-1],
                "hidden_size": hidden_size,
                "num_layers": LSTM_NUM_LAYERS,
                "num_classes": num_classes,
                "label_to_idx": dataset.label_to_idx,
                "idx_to_label": dataset.idx_to_label,
                "val_acc": val_acc,
                "epoch": epoch,
            }, output_path)
            logger.info("  ✓ New best model saved (val_acc=%.3f)", val_acc)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    logger.info("Training complete. Best Val Accuracy: %.3f", best_val_acc)
    logger.info("Model saved to: %s", output_path)

    # ── Export to ONNX for CPU-optimized inference ──────────
    logger.info("Exporting LSTM to ONNX...")
    _export_lstm_onnx(output_path)

    return best_val_acc


def _export_lstm_onnx(model_pt_path: str):
    """Export trained LSTM model to ONNX for CPU-optimized inference."""
    if not os.path.exists(model_pt_path):
        logger.error("Model not found at %s — cannot export ONNX.", model_pt_path)
        return
    try:
        ckpt = torch.load(model_pt_path, map_location="cpu")
        model = HARLSTMClassifier(
            feature_dim=ckpt["feature_dim"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            num_classes=ckpt["num_classes"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        seq_len = SEQUENCE_WINDOW
        feature_dim = ckpt["feature_dim"]
        dummy = torch.randn(1, seq_len, feature_dim)

        torch.onnx.export(
            model, dummy, LSTM_ONNX_PATH,
            export_params=True,
            opset_version=17,
            input_names=["skeleton_sequence"],
            output_names=["logits"],
            dynamic_axes={
                "skeleton_sequence": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
        logger.info("LSTM ONNX saved: %s", LSTM_ONNX_PATH)
    except Exception as e:
        logger.error("LSTM ONNX export failed: %s", e)


if __name__ == "__main__":
    train_model()
