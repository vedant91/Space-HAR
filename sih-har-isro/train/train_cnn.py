"""
Custom CNN Activity Classifier — Training Script
=================================================
Trains a custom lightweight 3D-aware CNN from scratch on RTX 3050 6GB.
Input:  N stacked consecutive frames (temporal window) → (B, C*N, H, W)
Output: Step class (0..num_steps)

Architecture: Custom ConvNet (no pretrained weights anywhere)
  ├── Stem: Conv 7×7 → BN → ReLU → MaxPool
  ├── Block 1: 2× (Conv3×3 → BN → ReLU) + residual
  ├── Block 2: 2× (Conv3×3 → BN → ReLU) + residual
  ├── Block 3: 2× (Conv3×3 → BN → ReLU) + residual  
  ├── Global Average Pooling
  └── Classifier Head: Linear → Dropout → Linear → Softmax

RTX 3050 6GB can train this with batch=32, fp16, in ~20 min.

Usage:
    python train/train_cnn.py
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

from torch.utils.data import Dataset, DataLoader
# torch.cuda.amp.{GradScaler,autocast} are deprecated in torch 2.x in favour
# of the device-generic torch.amp API; the old path emits a FutureWarning on
# every call and is scheduled for removal. Fall back for older torch.
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE_ARG = True
except ImportError:  # torch < 2.3
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE_ARG = False

_NUM_WORKERS = 0 if platform.system() == "Windows" else 4

# Force UTF-8 on stdout/stderr. A stock Windows console is cp1252, and both
# this file's progress banners and torch's own ONNX exporter emit non-ASCII
# (the exporter prints a check mark). Without this the export step dies with
# UnicodeEncodeError *after* a successful training run, so the .pt exists but
# the .onnx the pipeline prefers never appears.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    CNN_EPOCHS, CNN_BATCH_SIZE, CNN_IMG_SIZE, CNN_LEARNING_RATE,
    CNN_WEIGHT_DECAY, CNN_DROPOUT, CNN_NUM_FRAMES_IN,
    TRAIN_VAL_SPLIT, USE_AMP, CNN_MODEL_PATH, EXPERIMENT_STEPS, NUM_STEPS,
)

# NUM_STEPS is config's single source of truth for len(EXPERIMENT_STEPS)+1
# (+1 for idle) — this used to recompute it locally under the name
# NUM_CLASSES, which collided in meaning (not value — this file's is 9,
# config's own unrelated NUM_CLASSES was 4 HSV-detection classes) with
# anyone who later did `from config.experiment_config import NUM_CLASSES`
# in this file.
NUM_CLASSES = NUM_STEPS
IN_CHANNELS = 3 * CNN_NUM_FRAMES_IN       # RGB × N frames


# ═══════════════════════════════════════════════════════════════
# Model Architecture (100% from scratch)
# ═══════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    """Standard Conv → BN → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """
    Residual block with optional channel projection.
    Input → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+skip) → ReLU
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Skip connection: project if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return self.relu(out)


class HARActivityCNN(nn.Module):
    """
    Custom HAR Activity Classifier — trained from scratch.

    Takes a stack of N consecutive BGR frames and predicts the current
    experiment step being performed by the astronaut.

    Input:  (batch, 3*N_frames, H, W)  — N_frames stacked as channels
    Output: (batch, num_classes)

    Architecture overview:
      Stem [7×7] → [3 Residual Stages] → [GAP] → [Classifier]
    Total params: ~2.1M (tiny, fast, fits entirely in L2 cache)
    """

    def __init__(self, in_channels: int = IN_CHANNELS,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = CNN_DROPOUT):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 7×7 conv captures large spatial context for activity
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 32 ch, H/4 × W/4

        # Stage 1: 32 → 64 channels
        self.stage1 = nn.Sequential(
            ResBlock(32,  64, stride=1),
            ResBlock(64,  64, stride=1),
        )
        # 64 ch, H/4 × W/4

        # Stage 2: 64 → 128 channels (spatial halving)
        self.stage2 = nn.Sequential(
            ResBlock(64,  128, stride=2),
            ResBlock(128, 128, stride=1),
            ResBlock(128, 128, stride=1),
        )
        # 128 ch, H/8 × W/8

        # Stage 3: 128 → 256 channels (spatial halving)
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
        )
        # 256 ch, H/16 × W/16

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Initialize all weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class FrameStackDataset(Dataset):
    """
    Loads stacked-frame samples for CNN training.

    Each sample = N consecutive frames stacked as (C*N, H, W)
    Label = step_id (0-indexed class)

    Expects directory structure:
        dataset/annotated/
          step_01/ frame0001.jpg, frame0002.jpg, ...
          step_02/ ...
          ...
    """

    def __init__(self, data_dir: str, img_size: int = CNN_IMG_SIZE,
                 n_frames: int = CNN_NUM_FRAMES_IN, augment: bool = True,
                 split: str = "all", train_frac: float = TRAIN_VAL_SPLIT):
        """
        split: "all" (every window), "train" (windows built only from the
        first `train_frac` of each step folder's frames), or "val" (the
        remaining tail). Splitting the underlying FRAMES first — before
        windowing — means no window can straddle the train/val boundary and
        no frame is ever shared between a train and a val sample. Previously
        this class built one flat, stride-1 (87.5% frame overlap between
        adjacent samples) window list and let torch's random_split divide
        individual windows, which routinely put near-duplicate windows on
        both sides of the split.
        """
        self.img_size = img_size
        self.n_frames = n_frames
        self.augment  = augment
        self.samples  = []   # [(frame_paths_list, label_idx)]
        self.label_map = {}  # {step_id: class_idx}

        self._load_samples(data_dir, split=split, train_frac=train_frac)
        logger.info("FrameStackDataset(split=%s): %d samples, %d classes",
                   split, len(self.samples), len(self.label_map))

    def _load_samples(self, data_dir: str, split: str = "all", train_frac: float = 0.8):
        data_path = Path(data_dir)
        step_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

        for class_idx, step_dir in enumerate(step_dirs):
            # Extract step_id from folder name (step_01, step_02, ...)
            try:
                step_id = int(step_dir.name.split("_")[1])
            except (IndexError, ValueError):
                step_id = class_idx + 1
            self.label_map[step_id] = class_idx

            frames = sorted(
                list(step_dir.glob("*.jpg")) + list(step_dir.glob("*.png"))
            )
            if len(frames) < self.n_frames:
                continue

            split_at = int(len(frames) * train_frac)
            if split == "train":
                usable = frames[:split_at]
            elif split == "val":
                usable = frames[split_at:]
            else:
                usable = frames
            if len(usable) < self.n_frames:
                continue

            # Create sliding window samples within this split's frame range only
            for i in range(len(usable) - self.n_frames + 1):
                window = usable[i: i + self.n_frames]
                self.samples.append((window, class_idx))

    def _load_frame(self, path: Path) -> np.ndarray:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _augment(self, frames: list) -> list:
        """Apply consistent augmentation across all frames in a window.

        NOTE ON HORIZONTAL FLIP — deliberately absent.

        This used to apply `np.fliplr` with probability 0.5 while keeping the
        label. That is label-destroying for this protocol, not merely
        aggressive: the eight steps are defined partly by WHICH SIDE of the
        container the action happens on. Step 3 picks the red box from the
        left, step 6 picks the yellow box from the right; step 5 places into
        the left restraint zone, step 8 into the right. Mirroring the image
        maps step 3's geometry onto step 6's and step 5's onto step 8's while
        asserting the original label, so the network was being explicitly
        taught that left/right position carries no information - discarding
        the single most discriminative cue it has, and leaving colour as the
        only separator between those pairs.

        Rotation IS kept, including the full 90-degree steps: in microgravity
        the crew member genuinely has no fixed 'up', so a rotated frame is a
        real observation, not a fabricated one. Rotation preserves handedness;
        reflection does not.
        """
        import cv2
        # Random brightness shift
        shift = np.random.uniform(0.7, 1.3)
        frames = [np.clip(f.astype(np.float32) * shift, 0, 255).astype(np.uint8) for f in frames]
        # Random small rotation (±20°) — camera mounting tolerance / body roll
        angle = np.random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((self.img_size // 2, self.img_size // 2), angle, 1.0)
        frames = [cv2.warpAffine(f, M, (self.img_size, self.img_size)) for f in frames]
        # Orientation augmentation: random 0/90/180/270 rotation (microgravity)
        k = np.random.randint(0, 4)
        if k > 0:
            frames = [np.ascontiguousarray(np.rot90(f, k=k)) for f in frames]
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = [self._load_frame(p) for p in frame_paths]

        if self.augment:
            frames = self._augment(frames)

        # Stack frames as channels: (N, H, W, C) → (N*C, H, W)
        stacked = np.concatenate(frames, axis=-1)  # (H, W, N*3)
        stacked = stacked.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Normalize (mean/std from ImageNet-like stats, adapted for stacked frames)
        mean = np.array([0.485, 0.456, 0.406] * self.n_frames, dtype=np.float32).reshape(-1, 1, 1)
        std  = np.array([0.229, 0.224, 0.225] * self.n_frames, dtype=np.float32).reshape(-1, 1, 1)
        stacked = (stacked - mean) / (std + 1e-7)

        return torch.FloatTensor(stacked), torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════

def train_cnn(data_dir: str = "dataset/annotated",
              output_path: Optional[str] = None,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None):
    if output_path is None:
        output_path = CNN_MODEL_PATH
    epochs = CNN_EPOCHS if epochs is None else epochs
    batch_size = CNN_BATCH_SIZE if batch_size is None else batch_size

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────
    # Seeded so the split itself is reproducible run-to-run (previously
    # random_split had no generator at all).
    torch.manual_seed(42)
    train_ds = FrameStackDataset(data_dir, augment=True, split="train")
    val_ds   = FrameStackDataset(data_dir, augment=False, split="val")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"No samples found in {data_dir} for one or both splits. "
            "Run mediapipe_labeler first, or check TRAIN_VAL_SPLIT against clip length."
        )
    train_n, val_n = len(train_ds), len(val_ds)
    full_label_map = train_ds.label_map  # identical to val_ds.label_map (same folders)

    # Size the head to the classes that actually exist on disk, not to
    # config.NUM_STEPS. NUM_STEPS is len(EXPERIMENT_STEPS)+1 = 9 (the +1 being
    # an "idle" class), but `dataset/annotated/` only ever contains the eight
    # step_XX folders - there is no idle folder and nothing writes one. The
    # model therefore carried a ninth logit that no sample could ever
    # activate: dead capacity that softmax still had to normalise over,
    # slightly depressing every real class's confidence against
    # STEP_CONFIDENCE_THRESHOLD at inference.
    num_classes = len(full_label_map)
    if num_classes != NUM_CLASSES:
        logger.info("Sizing classifier head to %d classes found on disk "
                    "(config.NUM_STEPS=%d includes an idle class that %s does "
                    "not contain).", num_classes, NUM_CLASSES, data_dir)

    # A very short clip (frames_per_step * (1-TRAIN_VAL_SPLIT) < CNN_NUM_FRAMES_IN)
    # can leave a class with zero windows in one split's tail — silent unless
    # surfaced explicitly, and would skew val_acc without explaining why.
    train_classes = {c for _, c in train_ds.samples}
    val_classes = {c for _, c in val_ds.samples}
    missing_in_val = train_classes - val_classes
    if missing_in_val:
        logger.warning(
            "Classes with ZERO validation windows (clip too short for "
            "TRAIN_VAL_SPLIT=%.2f at n_frames=%d): class_idx=%s — val_acc will not "
            "reflect these classes. Increase frames_per_step or lower CNN_NUM_FRAMES_IN.",
            TRAIN_VAL_SPLIT, CNN_NUM_FRAMES_IN, sorted(missing_in_val))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=True,
        persistent_workers=_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | AMP: %s", device, USE_AMP and device.type == "cuda")

    model = HARActivityCNN(
        in_channels=IN_CHANNELS,
        num_classes=num_classes,
        dropout=CNN_DROPOUT,
    ).to(device)

    logger.info("Model parameters: %s", f"{model.count_parameters():,}")
    logger.info("Train: %d | Val: %d | Classes: %d", train_n, val_n, num_classes)

    # ── Optimizer / Scheduler ──────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    _amp_on = USE_AMP and device.type == "cuda"
    scaler = (GradScaler("cuda", enabled=_amp_on) if _AMP_DEVICE_ARG
              else GradScaler(enabled=_amp_on))

    # ── Training loop ──────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 12

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with (autocast("cuda", enabled=_amp_on) if _AMP_DEVICE_ARG
                  else autocast(enabled=_amp_on)):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        correct = total = val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with (autocast("cuda", enabled=_amp_on) if _AMP_DEVICE_ARG
                      else autocast(enabled=_amp_on)):
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item()
                preds    = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += len(y_batch)

        val_acc = correct / total if total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Ep %03d/%03d | TrainLoss=%.4f | ValLoss=%.4f | ValAcc=%.3f | LR=%.2e",
                epoch, epochs, avg_train_loss, avg_val_loss, val_acc,
                optimizer.param_groups[0]["lr"]
            )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state":    model.state_dict(),
                "in_channels":    IN_CHANNELS,
                "num_classes":    num_classes,
                "img_size":       CNN_IMG_SIZE,
                "n_frames_in":    CNN_NUM_FRAMES_IN,
                "label_map":      full_label_map,
                "val_acc":        val_acc,
                "epoch":          epoch,
                "architecture":   "HARActivityCNN-scratch",
            }, output_path)
            logger.info("  ✓ Best model saved (val_acc=%.3f)", val_acc)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info("Early stopping at epoch %d (best=%.3f).", epoch, best_val_acc)
                break

    # Export to ONNX
    logger.info("Exporting to ONNX...")
    _export_onnx(output_path)
    logger.info("Training complete. Best Val Acc: %.3f", best_val_acc)
    return best_val_acc


def _export_onnx(model_pt_path: str):
    """Export trained model to ONNX for CPU-optimized inference."""
    from config.experiment_config import CNN_ONNX_PATH
    try:
        ckpt  = torch.load(model_pt_path, map_location="cpu", weights_only=False)
        model = HARActivityCNN(
            in_channels=ckpt["in_channels"],
            num_classes=ckpt["num_classes"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        dummy = torch.randn(1, ckpt["in_channels"], ckpt["img_size"], ckpt["img_size"])
        torch.onnx.export(
            model, dummy, CNN_ONNX_PATH,
            export_params=True,
            opset_version=17,
            input_names=["frames"],
            output_names=["logits"],
            dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
        )
        logger.info("ONNX saved: %s", CNN_ONNX_PATH)
    except Exception as e:
        logger.error("ONNX export failed: %s", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Custom CNN from Scratch")
    parser.add_argument("--data", default="dataset/annotated",
                        help="Data directory with step_XX folders")
    parser.add_argument("--output", default=None, help="Output model path")
    args = parser.parse_args()
    train_cnn(args.data, args.output)
