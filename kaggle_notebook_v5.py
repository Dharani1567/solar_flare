# -*- coding: utf-8 -*-
"""
==========================================================================
  SOLAR FLARE FORECASTING - COMPLETE KAGGLE NOTEBOOK (v5 Dataset)
  Paste this as a SINGLE CELL in your Kaggle Notebook.
  Works with dataset: solar-flare-package-v
  Files expected in /kaggle/input/:
    dataset_v5/X_sequences_v5.npy
    dataset_v5/y_labels_v5.npy
    dataset_v5/sequence_metadata_v5.csv
    dataset_v5/dataset_statistics_v5.json
    README.md
    fold_assignments.csv
    kaggle_train_groupkfold.py
    models.py
    requirements.txt
    train_all_models.py
==========================================================================
"""

# -----------------------------------------------------------------------
# CELL 1: SETUP - Auto-detect dataset path and install dependencies
# -----------------------------------------------------------------------
import os
import sys
import shutil
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("  SOLAR FLARE FORECASTING - KAGGLE GPU BENCHMARK")
print("=" * 60)

# -------- Step 1: Auto-locate all files from /kaggle/input --------
print("\n[SETUP] Searching for dataset files in /kaggle/input...")

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# Recursively find X_sequences_v5.npy
x_file = None
y_file = None
meta_file = None
stats_file = None
dataset_dir = None

for root, dirs, files in os.walk(KAGGLE_INPUT):
    for f in files:
        full = os.path.join(root, f)
        if "X_sequences_v5" in f:
            x_file = full
            dataset_dir = root
        elif "y_labels_v5" in f:
            y_file = full
        elif "sequence_metadata_v5" in f:
            meta_file = full
        elif "dataset_statistics_v5" in f:
            stats_file = full

# Find script files at top-level of the input dataset
script_dir = None
for root, dirs, files in os.walk(KAGGLE_INPUT):
    if "train_all_models.py" in files or "models.py" in files:
        script_dir = root
        break

if not x_file:
    # Fallback: look for any X.npy / y.npy files
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        for f in files:
            full = os.path.join(root, f)
            if f == "X.npy":
                x_file = full
                dataset_dir = root
            elif f == "y.npy":
                y_file = full
            elif "metadata" in f.lower() and f.endswith(".csv"):
                meta_file = full

print(f"[SETUP] X file found     : {x_file}")
print(f"[SETUP] y file found     : {y_file}")
print(f"[SETUP] Metadata found   : {meta_file}")
print(f"[SETUP] Scripts found at : {script_dir}")

# -------- Step 2: Copy scripts to /kaggle/working --------
if script_dir and script_dir != KAGGLE_WORKING:
    for item in os.listdir(script_dir):
        src = os.path.join(script_dir, item)
        dst = os.path.join(KAGGLE_WORKING, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"[SETUP] Copied {item}")
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"[SETUP] Copied directory: {item}")

# -------- Step 3: Load the dataset --------
print("\n[DATA] Loading dataset...")

if not x_file or not y_file:
    raise FileNotFoundError(
        "Could not find X and y files in /kaggle/input.\n"
        "Please make sure you added the dataset via '+ Add Data' on the right panel.\n"
        f"Contents of /kaggle/input: {os.listdir(KAGGLE_INPUT)}"
    )

X = np.load(x_file)
y = np.load(y_file)

if meta_file:
    metadata = pd.read_csv(meta_file)
else:
    metadata = pd.DataFrame({"sample_id": range(len(y)), "label": y})

# Auto-detect group column
group_col = None
for col in ["flare_event_id", "sequence_id", "date", "sample_id"]:
    if col in metadata.columns:
        group_col = col
        break

if group_col:
    groups = metadata[group_col].values
else:
    groups = np.arange(len(y))

print(f"[DATA] X shape           : {X.shape}")
print(f"[DATA] y shape           : {y.shape}")
print(f"[DATA] Positive samples  : {np.sum(y == 1)}")
print(f"[DATA] Negative samples  : {np.sum(y == 0)}")
print(f"[DATA] Grouping column   : {group_col}")
print(f"[DATA] Unique groups     : {len(np.unique(groups))}")

if stats_file:
    with open(stats_file) as f:
        stats = json.load(f)
    print(f"[DATA] Dataset stats     : {stats}")

# -------- Step 4: GPU Check --------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, roc_curve
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[GPU] Computation device : {device}")
if device.type == "cuda":
    print(f"[GPU] GPU Name           : {torch.cuda.get_device_name(0)}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------- Step 5: Create output directories --------
os.makedirs(os.path.join(KAGGLE_WORKING, "plots"), exist_ok=True)
os.makedirs(os.path.join(KAGGLE_WORKING, "results"), exist_ok=True)
PLOTS_DIR = os.path.join(KAGGLE_WORKING, "plots")
RESULTS_DIR = os.path.join(KAGGLE_WORKING, "results")

# ===================================================================
# MODEL DEFINITIONS
# ===================================================================

class Conv1DModel(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveMaxPool1d(16)
        self.fc1 = nn.Linear(64 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class CNNBiLSTMModel(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.max_pool1d(x, 4)
        x = x.transpose(1, 2)
        out, (hn, _) = self.lstm(x)
        last = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(last)


class CNNAttentionBiLSTMModel(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(64, 1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.max_pool1d(x, 4)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return self.fc(context)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)


class TCNModel(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.b1 = TCNBlock(in_channels, 32, dilation=1)
        self.b2 = TCNBlock(32, 64, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return self.fc(self.pool(x).squeeze(-1))


class InceptionModule1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        ch = out_ch // 4
        self.p1 = nn.Conv1d(in_ch, ch, 1)
        self.p2 = nn.Conv1d(in_ch, ch, 3, padding=1)
        self.p3 = nn.Conv1d(in_ch, ch, 5, padding=2)
        self.p4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, ch, 1)
        )

    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)], dim=1)


class InceptionTimeModel(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.inc1 = InceptionModule1D(in_channels, 32)
        self.inc2 = InceptionModule1D(32, 64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.inc1(x)
        x = self.inc2(x)
        return self.fc(self.pool(x).squeeze(-1))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        x = F.relu(self.bn1(self.c1(x)))
        x = self.bn2(self.c2(x))
        return F.relu(x + res)


class ResNet1DModel(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.b1 = ResBlock1D(in_channels, 32)
        self.b2 = ResBlock1D(32, 64)
        self.b3 = ResBlock1D(64, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = F.max_pool1d(x, 2)
        x = self.b2(x)
        x = F.max_pool1d(x, 2)
        x = self.b3(x)
        return self.fc(self.pool(x).squeeze(-1))


# ===================================================================
# TRAINING LOOP
# ===================================================================

# Downsample on CPU to keep runtime acceptable
is_cpu = (device.type == "cpu")
if is_cpu:
    print("\n[INFO] CPU mode: downsampling time axis 3600 -> 360 for speed.")
    X_use = X[:, ::10, :]
else:
    print("\n[INFO] GPU mode: using full 3600-step sequences.")
    X_use = X

num_channels = X_use.shape[2]
EPOCHS = 8 if is_cpu else 30
BATCH = 32

print(f"[INFO] Epochs per fold: {EPOCHS}")

MODELS = {
    "1D CNN": Conv1DModel,
    "CNN + BiLSTM": CNNBiLSTMModel,
    "CNN + Attention + BiLSTM": CNNAttentionBiLSTMModel,
    "TCN": TCNModel,
    "InceptionTime": InceptionTimeModel,
    "ResNet1D": ResNet1DModel,
}

gkf = GroupKFold(n_splits=5)
results = []
oof_probs_dict = {}
oof_preds_dict = {}
loss_hist = {}

# ---- Sample time-series plot (before training) ----
print("\n[PLOT 1/8] Sample Solar Flare Time-Series...")
pos_idx = np.where(y == 1)[0][0]
neg_idx = np.where(y == 0)[0][0]
ch_names = ["Ch1 Soft X-ray", "Ch2 Hard X-ray", "Ch3 Count Rate", "Ch4 SoLEXS Peak"]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for c in range(num_channels):
    ax = axes[c // 2, c % 2]
    ax.plot(X[pos_idx, :, c], color="crimson", alpha=0.85, label="Flare (positive)")
    ax.plot(X[neg_idx, :, c], color="navy", alpha=0.65, linestyle="--", label="Quiet Sun (negative)")
    ax.set_title(ch_names[c], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step (s)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
plt.suptitle("Representative Solar Flare Telemetry Windows", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sample_solar_flare_series.png"), dpi=200)
plt.close()
print("   Saved sample_solar_flare_series.png")

# ---- Training ----
for name, ModelClass in MODELS.items():
    print(f"\n{'='*55}")
    print(f"  [TRAINING] {name}")
    print(f"{'='*55}")
    t0 = time.time()

    oof_probs = np.zeros(len(y))
    tr_losses_all, val_losses_all = [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_use, y, groups)):
        print(f"   Fold {fold+1}/5 ...", end=" ", flush=True)

        # Per-fold normalization
        mu = X_use[tr_idx].mean(axis=(0, 1), keepdims=True)
        sd = X_use[tr_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        X_tr = (X_use[tr_idx] - mu) / sd
        X_vl = (X_use[val_idx] - mu) / sd

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).transpose(1, 2).to(device)
        y_tr_t = torch.tensor(y[tr_idx], dtype=torch.long).to(device)
        X_vl_t = torch.tensor(X_vl, dtype=torch.float32).transpose(1, 2).to(device)
        y_vl_t = torch.tensor(y[val_idx], dtype=torch.long).to(device)

        tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH, shuffle=True)
        vl_loader = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=BATCH, shuffle=False)

        set_seed(42 + fold)
        model = ModelClass(in_channels=num_channels, num_classes=2).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()

        best_vloss = float("inf")
        best_state = None
        tr_losses, vl_losses = [], []

        for ep in range(EPOCHS):
            # --- Train ---
            model.train()
            tl, tc = 0.0, 0
            for bx, by in tr_loader:
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
                tl += loss.item() * len(by)
                tc += len(by)
            tr_losses.append(tl / tc)

            # --- Validate ---
            model.eval()
            vl, vc = 0.0, 0
            with torch.no_grad():
                for bx, by in vl_loader:
                    loss = crit(model(bx), by)
                    vl += loss.item() * len(by)
                    vc += len(by)
            vl /= vc
            vl_losses.append(vl)

            if vl < best_vloss:
                best_vloss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best weights
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Save fold-0 weights
        if fold == 0:
            fname = name.lower().replace(" ", "_").replace("+", "_").replace("/", "_") + "_best.pth"
            torch.save(best_state, os.path.join(RESULTS_DIR, fname))

        # OOF predictions
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for bx, _ in vl_loader:
                probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                fold_probs.extend(probs)

        oof_probs[val_idx] = np.array(fold_probs)
        tr_losses_all.append(tr_losses)
        vl_losses_all = vl_losses  # last fold for display
        print(f"val_loss={best_vloss:.4f}")

    elapsed = time.time() - t0
    oof_preds = (oof_probs >= 0.5).astype(int)

    acc  = accuracy_score(y, oof_preds)
    prec = precision_score(y, oof_preds, zero_division=0)
    rec  = recall_score(y, oof_preds, zero_division=0)
    f1   = f1_score(y, oof_preds, zero_division=0)
    rauc = roc_auc_score(y, oof_probs)
    p_, r_, _ = precision_recall_curve(y, oof_probs)
    prauc = auc(r_, p_)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {rauc:.4f}")
    print(f"  PR-AUC    : {prauc:.4f}")
    print(f"  Time      : {elapsed:.1f}s")

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1 Score": f1, "ROC-AUC": rauc,
        "PR-AUC": prauc, "Time (s)": round(elapsed, 1)
    })

    oof_probs_dict[name] = oof_probs
    oof_preds_dict[name] = oof_preds
    loss_hist[name] = (
        np.mean(tr_losses_all, axis=0).tolist(),
        vl_losses_all
    )

# ===================================================================
# SAVE BENCHMARK CSV
# ===================================================================
df_res = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
df_res.to_csv(csv_path, index=False)
print(f"\n[INFO] Saved benchmark_results.csv")
print(df_res.to_string(index=False))

# ===================================================================
# GENERATE ALL 7 REMAINING PLOTS
# ===================================================================

# 2. Training Loss Curves
print("\n[PLOT 2/8] Training Loss Curves...")
plt.figure(figsize=(10, 6))
for name, (trl, _) in loss_hist.items():
    plt.plot(range(1, len(trl)+1), trl, linewidth=2, label=name)
plt.title("Training Loss Curves", fontsize=13, fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "training_loss_curves.png"), dpi=200)
plt.close()

# 3. Validation Loss Curves
print("[PLOT 3/8] Validation Loss Curves...")
plt.figure(figsize=(10, 6))
for name, (_, vll) in loss_hist.items():
    plt.plot(range(1, len(vll)+1), vll, linewidth=2, linestyle="--", label=name)
plt.title("Validation Loss Curves", fontsize=13, fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Validation Loss")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "validation_loss_curves.png"), dpi=200)
plt.close()

# 4. ROC Curves
print("[PLOT 4/8] ROC Curves...")
plt.figure(figsize=(9, 7))
for name, probs in oof_probs_dict.items():
    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, probs):.4f})")
plt.plot([0,1],[0,1],"k--", alpha=0.5, label="Random")
plt.title("ROC Curves", fontsize=13, fontweight="bold")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=200)
plt.close()

# 5. Precision-Recall Curves
print("[PLOT 5/8] Precision-Recall Curves...")
plt.figure(figsize=(9, 7))
for name, probs in oof_probs_dict.items():
    p_, r_, _ = precision_recall_curve(y, probs)
    plt.plot(r_, p_, lw=2, label=f"{name} (AUC={auc(r_, p_):.4f})")
plt.title("Precision-Recall Curves", fontsize=13, fontweight="bold")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.legend(loc="lower left"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curves.png"), dpi=200)
plt.close()

# 6. Confusion Matrices
print("[PLOT 6/8] Confusion Matrices...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for idx, (name, preds) in enumerate(oof_preds_dict.items()):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticklabels(["Quiet", "Flare"]); ax.set_yticklabels(["Quiet", "Flare"])
plt.suptitle("Confusion Matrices - All 6 Models", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=200)
plt.close()

# 7. F1 Score Comparison Bar Chart
print("[PLOT 7/8] F1 Score Comparison...")
plt.figure(figsize=(10, 6))
colors = sns.color_palette("viridis", len(df_res))
bars = plt.bar(df_res["Model"], df_res["F1 Score"], color=colors, edgecolor="black", alpha=0.85)
plt.title("Model Comparison - F1 Score", fontsize=13, fontweight="bold")
plt.ylabel("F1 Score"); plt.ylim(0.75, 1.01)
plt.xticks(rotation=20, ha="right")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 0.005,
             f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "f1_score_comparison.png"), dpi=200)
plt.close()

# 8. ROC-AUC Comparison Bar Chart
print("[PLOT 8/8] ROC-AUC Comparison...")
plt.figure(figsize=(10, 6))
colors = sns.color_palette("magma", len(df_res))
bars = plt.bar(df_res["Model"], df_res["ROC-AUC"], color=colors, edgecolor="black", alpha=0.85)
plt.title("Model Comparison - ROC-AUC", fontsize=13, fontweight="bold")
plt.ylabel("ROC-AUC"); plt.ylim(0.85, 1.01)
plt.xticks(rotation=20, ha="right")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 0.003,
             f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_auc_comparison.png"), dpi=200)
plt.close()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 60)
print("  COMPLETE! All files saved in /kaggle/working/")
print("=" * 60)
print("\nSaved Plots:")
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"  plots/{f}")
print("\nSaved Results:")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  results/{f}")
print("\nFinal Benchmark:")
print(df_res[["Model","Accuracy","F1 Score","ROC-AUC","Time (s)"]].to_string(index=False))
