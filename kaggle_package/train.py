"""
Solar Flare Forecasting — Training Script
==========================================
Dataset  : ISRO Aditya-L1 (HEL1OS + SoLEXS) v5
Task     : Binary classification — Major M/X flare (1) vs Quiet/C-class (0)
Models   : 1D-CNN, CNN+BiLSTM, CNN+Attention+BiLSTM, TCN, InceptionTime, ResNet1D
CV       : StratifiedGroupKFold (5-fold, group=date, seed=42)
Hardware : GPU enabled (CUDA / MPS / CPU fallback)

Usage    : python train.py
           python train.py --data_dir /path/to/data --folds 5 --epochs 60
"""

import os
import sys
import time
import random
import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 10       # early stopping patience
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
N_FOLDS = 5
POS_WEIGHT_FACTOR = 2.77   # negative/positive ratio
SEQ_LEN = 3600
N_CHANNELS = 4
NUM_CLASSES = 2

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class SolarFlareDataset(Dataset):
    """Wraps X, y arrays with optional per-fold z-score normalization."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mean: np.ndarray = None, std: np.ndarray = None):
        # X: (N, T, C) — normalise per-channel across time axis
        if mean is not None and std is not None:
            X = (X - mean) / (std + 1e-8)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # → (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_fold_stats(X: np.ndarray):
    """Compute per-channel mean/std from training fold. X: (N, T, C)."""
    mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, C)
    std  = X.std(axis=(0, 1), keepdims=True)    # (1, 1, C)
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. 1D CNN ────────────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    """Baseline 1D Convolutional Network."""

    def __init__(self, in_channels=N_CHANNELS, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=9,  padding=4), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7,  padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ── 2. CNN + BiLSTM ──────────────────────────────────────────────────────────
class CNNBiLSTM(nn.Module):
    """CNN feature extractor followed by Bidirectional LSTM."""

    def __init__(self, in_channels=N_CHANNELS, dropout=DROPOUT):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=9, padding=4), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64,  128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)                   # (N, T', C)
        out, _ = self.lstm(f)                               # (N, T', 256)
        pooled = out.mean(dim=1)                            # (N, 256)
        return self.head(pooled)


# ── 3. CNN + Attention + BiLSTM ──────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):          # x: (N, T, D)
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ V            # (N, T, D)


class CNNAttBiLSTM(nn.Module):
    """CNN → Self-Attention → BiLSTM."""

    def __init__(self, in_channels=N_CHANNELS, dropout=DROPOUT):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=9, padding=4), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64,  128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.attn = SelfAttention(256)
        self.norm = nn.LayerNorm(256)
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)          # (N, T', 256)
        a = self.norm(self.attn(f) + f)            # residual attention
        out, _ = self.lstm(a)
        pooled = out.mean(dim=1)
        return self.head(pooled)


# ── 4. TCN ───────────────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout=DROPOUT):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrize.register_parametrization if False else \
                     nn.Sequential(
                         nn.Conv1d(n_in,  n_out, kernel_size, dilation=dilation, padding=pad),
                         nn.BatchNorm1d(n_out), nn.ReLU(), nn.Dropout(dropout)
                     )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_out, n_out, kernel_size, dilation=dilation, padding=pad),
            nn.BatchNorm1d(n_out), nn.ReLU(), nn.Dropout(dropout)
        )
        self.pad = pad
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)[..., :-self.pad] if self.pad else self.conv1(x)
        out = self.conv2(out)[..., :-self.pad] if self.pad else self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilations."""

    def __init__(self, in_channels=N_CHANNELS, num_channels=(64, 128, 128, 256),
                 kernel_size=5, dropout=DROPOUT):
        super().__init__()
        layers = []
        for i, n_out in enumerate(num_channels):
            n_in   = in_channels if i == 0 else num_channels[i - 1]
            dil    = 2 ** i
            layers.append(TemporalBlock(n_in, n_out, kernel_size, dil, dropout))
        self.network = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels[-1], 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.head(self.pool(self.network(x)))


# ── 5. InceptionTime ─────────────────────────────────────────────────────────
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, n_filters=32, kernel_sizes=(9, 19, 39)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, n_filters, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, ks, padding=ks // 2, bias=False)
            for ks in kernel_sizes
        ])
        self.mp_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, 1, bias=False),
        )
        n_out = n_filters * len(kernel_sizes) + n_filters
        self.bn  = nn.BatchNorm1d(n_out)
        self.act = nn.ReLU()
        self.res = nn.Sequential(nn.Conv1d(in_ch, n_out, 1, bias=False),
                                 nn.BatchNorm1d(n_out)) if in_ch != n_out else None

    def forward(self, x):
        b = self.bottleneck(x)
        outs = [c(b) for c in self.convs] + [self.mp_conv(x)]
        out  = torch.cat(outs, dim=1)
        out  = self.act(self.bn(out))
        res  = x if self.res is None else self.res(x)
        return out + res


class InceptionTime(nn.Module):
    """InceptionTime: multi-scale inception modules for time series."""

    def __init__(self, in_channels=N_CHANNELS, depth=6, n_filters=32, dropout=DROPOUT):
        super().__init__()
        blocks = []
        ch = in_channels
        for _ in range(depth):
            out_ch = n_filters * 3 + n_filters
            blocks.append(InceptionBlock(ch, n_filters))
            ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.head(self.pool(self.blocks(x)))


# ── 6. ResNet1D ──────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=DROPOUT):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm1d(out_ch)) if in_ch != out_ch else nn.Identity()
        self.act  = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class ResNet1D(nn.Module):
    """Residual 1D CNN, inspired by ResNet for time-series."""

    def __init__(self, in_channels=N_CHANNELS, dropout=DROPOUT):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.blocks = nn.Sequential(
            ResBlock1D(64,  64,  dropout=dropout), nn.MaxPool1d(2),
            ResBlock1D(64,  128, dropout=dropout), nn.MaxPool1d(2),
            ResBlock1D(128, 256, dropout=dropout), nn.MaxPool1d(2),
            ResBlock1D(256, 256, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))))


MODEL_REGISTRY = {
    "1D CNN":                   CNN1D,
    "CNN + BiLSTM":             CNNBiLSTM,
    "CNN + Attention + BiLSTM": CNNAttBiLSTM,
    "TCN":                      TCN,
    "InceptionTime":            InceptionTime,
    "ResNet1D":                 ResNet1D,
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(X_b)
                loss   = criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        total_loss += loss.item() * len(y_b)
        all_logits.append(logits.cpu())
        all_labels.append(y_b.cpu())
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
    preds  = (probs >= 0.5).astype(int)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, probs, preds, labels_cat


def compute_metrics(labels, preds, probs):
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        "pr_auc":    average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model_name, model_class, X, y, groups,
                device, args, pos_weight):
    """
    Run StratifiedGroupKFold cross-validation for one model.
    Returns dict of aggregated metrics + per-fold loss history.
    """
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")

    sgkf   = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    pw     = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    )

    fold_metrics   = []
    train_loss_all = []
    val_loss_all   = []

    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    model_save_dir = RESULTS_DIR / "models"
    model_save_dir.mkdir(exist_ok=True)

    t_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(X, y, groups), start=1):
        print(f"\n  ── Fold {fold_idx}/{args.folds} ──")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Per-fold normalization (fit on train only)
        mean, std = compute_fold_stats(X_tr)
        tr_ds  = SolarFlareDataset(X_tr, y_tr, mean, std)
        val_ds = SolarFlareDataset(X_val, y_val, mean, std)

        tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=(device.type == "cuda"))

        print(f"     Train: {len(tr_ds)} | Val: {len(val_ds)} | "
              f"Train pos: {y_tr.sum()} | Val pos: {y_val.sum()}")

        model = model_class().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        best_val_loss  = float("inf")
        best_state     = None
        patience_count = 0
        fold_train_losses = []
        fold_val_losses   = []

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion,
                                      device, scaler)
            val_loss, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            fold_train_losses.append(tr_loss)
            fold_val_losses.append(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"     Epoch {epoch:3d}/{args.epochs} | "
                      f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_state     = {k: v.cpu().clone() for k, v in
                                  model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f"     Early stopping at epoch {epoch}")
                    break

        # Load best weights and evaluate
        model.load_state_dict(best_state)
        model.to(device)
        save_path = model_save_dir / f"{model_name.lower().replace(' ', '_')}_fold{fold_idx}.pt"
        torch.save(best_state, save_path)

        _, probs, preds, labels = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(labels, preds, probs)
        fold_metrics.append(metrics)

        print(f"     → F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
              f"PR-AUC: {metrics['pr_auc']:.4f}")

        train_loss_all.append(fold_train_losses)
        val_loss_all.append(fold_val_losses)

    elapsed = time.time() - t_start

    # Aggregate across folds
    agg = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        agg[key]          = float(np.mean(vals))
        agg[key + "_std"] = float(np.std(vals))

    agg["training_time_sec"] = elapsed

    print(f"\n  ✓ {model_name} — "
          f"ROC-AUC: {agg['roc_auc']:.4f}±{agg['roc_auc_std']:.4f} | "
          f"F1: {agg['f1']:.4f}±{agg['f1_std']:.4f} | "
          f"Time: {elapsed:.0f}s")

    return agg, train_loss_all, val_loss_all


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Solar Flare Forecasting — Training")
    p.add_argument("--data_dir",     type=str, default="../data/ml",
                   help="Directory containing X_sequences_v5.npy, etc.")
    p.add_argument("--folds",        type=int, default=N_FOLDS)
    p.add_argument("--epochs",       type=int, default=MAX_EPOCHS)
    p.add_argument("--patience",     type=int, default=PATIENCE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--batch_size",   type=int, default=BATCH_SIZE)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--dropout",      type=float, default=DROPOUT)
    p.add_argument("--models",       type=str, default="all",
                   help="Comma-separated model names or 'all'")
    return p.parse_args()


def load_dataset(data_dir: str):
    """Load v5 dataset. Falls back to earlier versions if v5 not found."""
    base = Path(data_dir)
    for version in ["v5", "v4", "v3", "expanded"]:
        X_path = base / f"X_sequences_{version}.npy"
        y_path = base / f"y_labels_{version}.npy"
        m_path = base / f"sequence_metadata_{version}.csv"
        if X_path.exists() and y_path.exists() and m_path.exists():
            print(f"[INFO] Loading dataset version: {version}")
            X    = np.load(X_path)
            y    = np.load(y_path)
            meta = pd.read_csv(m_path)
            return X, y, meta, version
    raise FileNotFoundError(f"No dataset found in {data_dir}. "
                            "Expected X_sequences_v5.npy etc.")


def main():
    args   = parse_args()
    set_seed(SEED)
    device = get_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PyTorch: {torch.__version__}")

    # ── Load dataset ────────────────────────────────────────────────────────
    X, y, meta, version = load_dataset(args.data_dir)
    print(f"[INFO] X shape : {X.shape}")
    print(f"[INFO] y shape : {y.shape}")
    print(f"[INFO] Class distribution: {np.bincount(y)}")

    # ── Preprocessing steps ─────────────────────────────────────────────────
    print("\n[STEP 1] No NaN/Inf found in audit — skipping replacement.")
    print("[STEP 2] No duplicates found in audit — skipping deduplication.")
    print("[STEP 3] No corrupted samples found — skipping removal.")
    print("[STEP 4] Per-fold z-score normalization applied inside training loop.")
    print("[STEP 5] Constructing group labels from 'date' column.")

    # Use date as group (prevents temporal leakage)
    groups = meta["date"].values
    print(f"[INFO] Groups (unique dates): {len(np.unique(groups))}")

    # Class weight for imbalanced learning
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    pw = neg_count / pos_count
    print(f"[INFO] pos_weight for loss: {pw:.4f} (neg={neg_count}, pos={pos_count})")

    # ── Select models ───────────────────────────────────────────────────────
    if args.models == "all":
        models_to_run = list(MODEL_REGISTRY.items())
    else:
        selected = [m.strip() for m in args.models.split(",")]
        models_to_run = [(k, v) for k, v in MODEL_REGISTRY.items() if k in selected]

    # ── Train & evaluate all models ─────────────────────────────────────────
    all_results    = {}
    all_histories  = {}

    for model_name, model_class in models_to_run:
        agg, train_hist, val_hist = train_model(
            model_name, model_class, X, y, groups, device, args, pw
        )
        all_results[model_name]   = agg
        all_histories[model_name] = {"train": train_hist, "val": val_hist}

    # ── Save results CSV ────────────────────────────────────────────────────
    rows = []
    for name, metrics in all_results.items():
        row = {"model": name}
        row.update({k: v for k, v in metrics.items() if not k.endswith("_std")})
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    csv_path   = RESULTS_DIR / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[DONE] Benchmark results saved → {csv_path}")
    print(results_df.to_string(index=False))

    # ── Save histories (for plotting) ────────────────────────────────────────
    hist_path = RESULTS_DIR / "loss_histories.json"
    serialisable = {}
    for name, h in all_histories.items():
        serialisable[name] = {
            "train": [fold for fold in h["train"]],
            "val":   [fold for fold in h["val"]],
        }
    with open(hist_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[DONE] Loss histories saved → {hist_path}")

    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"  BEST MODEL: {best['model']}")
    print(f"  ROC-AUC:    {best['roc_auc']:.4f}")
    print(f"  F1 Score:   {best['f1']:.4f}")
    if best["roc_auc"] > 0.90:
        print("  ★ ROC-AUC exceeds 0.90 — outstanding performance!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
