"""
Solar Flare Forecasting — Evaluation & Plotting Script
=======================================================
Generates all benchmark plots after training is complete.
Run after train.py has produced:
  results/benchmark_results.csv
  results/loss_histories.json

Usage: python evaluate.py
       python evaluate.py --data_dir ../data/ml
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Import model definitions from train.py
# ─────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    MODEL_REGISTRY, SolarFlareDataset, compute_fold_stats,
    compute_metrics, evaluate, set_seed, get_device, load_dataset,
    SEED, N_FOLDS, BATCH_SIZE, POS_WEIGHT_FACTOR
)

RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Consistent color palette for models
PALETTE = {
    "1D CNN":                   "#E63946",
    "CNN + BiLSTM":             "#457B9D",
    "CNN + Attention + BiLSTM": "#2A9D8F",
    "TCN":                      "#E9C46A",
    "InceptionTime":            "#F4A261",
    "ResNet1D":                 "#264653",
}


def parse_args():
    p = argparse.ArgumentParser(description="Solar Flare Forecasting — Evaluate & Plot")
    p.add_argument("--data_dir",   type=str, default="../data/ml")
    p.add_argument("--folds",      type=int, default=N_FOLDS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Sample Solar Flare Time Series
# ─────────────────────────────────────────────────────────────────────────────
def plot_sample_sequences(X, y, save_path=PLOTS_DIR / "sample_flare_sequences.png"):
    """Show one positive and one negative sample across all 4 channels."""
    pos_idx = np.where(y == 1)[0][0]
    neg_idx = np.where(y == 0)[0][0]
    channel_names = [
        "CdTe 1 (10–20 keV)", "CdTe 2 (20–50 keV)",
        "CZT 1 (50–100 keV)", "CZT2/SoLEXS (100–150 keV)"
    ]
    t = np.arange(3600) / 60  # minutes

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle("Sample Solar Flare Time Series — Aditya-L1 (HEL1OS + SoLEXS)",
                 fontsize=14, fontweight="bold", y=1.01)

    for ch in range(4):
        # Major flare (positive)
        ax = axes[ch, 0]
        ax.plot(t, X[pos_idx, :, ch], color="#E63946", linewidth=0.8)
        ax.set_ylabel("Counts (log-scaled)", fontsize=8)
        ax.set_xlabel("Time (min)", fontsize=8)
        if ch == 0:
            ax.set_title("Major M/X Flare (label=1)", fontsize=11, color="#E63946",
                         fontweight="bold")
        ax.annotate(channel_names[ch], xy=(0.02, 0.93), xycoords="axes fraction",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.grid(True, alpha=0.3)

        # Quiet sun (negative)
        ax = axes[ch, 1]
        ax.plot(t, X[neg_idx, :, ch], color="#457B9D", linewidth=0.8)
        ax.set_xlabel("Time (min)", fontsize=8)
        if ch == 0:
            ax.set_title("Quiet / C-class Event (label=0)", fontsize=11, color="#457B9D",
                         fontweight="bold")
        ax.annotate(channel_names[ch], xy=(0.02, 0.93), xycoords="axes fraction",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Training & Validation Loss Curves (per model, fold 1)
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(histories: dict):
    """Plot train/val loss curves for each model (fold 1 shown)."""
    n_models = len(histories)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    fig.suptitle("Training & Validation Loss Curves (Fold 1)",
                 fontsize=14, fontweight="bold")

    for i, (name, h) in enumerate(histories.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        color = PALETTE.get(name, "#333333")
        tr = h["train"][0] if h["train"] else []
        vl = h["val"][0]   if h["val"]   else []
        epochs = list(range(1, len(tr) + 1))
        ax.plot(epochs, tr, label="Train", color=color, linewidth=1.5)
        ax.plot(epochs, vl, label="Val",   color=color, linewidth=1.5, linestyle="--")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(all_probs: dict):
    """all_probs: {model_name: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

    for name, (y_true, y_prob) in all_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score   = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE.get(name, "#333333"), lw=2,
                label=f"{name} (AUC={auc_score:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_pr_curves(all_probs: dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Baseline: positive class ratio
    baseline = None

    for name, (y_true, y_prob) in all_probs.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap            = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, color=PALETTE.get(name, "#333333"), lw=2,
                label=f"{name} (AP={ap:.3f})")
        if baseline is None:
            baseline = y_true.mean()

    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Random (AP={baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "pr_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(all_preds: dict):
    """all_preds: {model_name: (y_true, y_pred)}"""
    n_models = len(all_preds)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("Confusion Matrices — All Models (StratifiedGroupKFold OOF)",
                 fontsize=13, fontweight="bold")
    labels = ["Quiet/C\n(0)", "Major\nFlare (1)"]

    for i, (name, (y_true, y_pred)) in enumerate(all_preds.items()):
        if i >= len(axes):
            break
        cm = confusion_matrix(y_true, y_pred)
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        ax.set_title(f"{name}\nF1={f1:.3f}", fontsize=9, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=8)
        ax.set_xlabel("Predicted Label", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — F1 Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_f1_comparison(results_df: pd.DataFrame):
    df = results_df.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE.get(n, "#333333") for n in df["model"]]
    bars = ax.barh(df["model"], df["f1"], color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, df["f1"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("F1 Score (5-fold CV mean)", fontsize=11)
    ax.set_title("F1 Score Comparison — All Models", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5, label="F1=0.50 baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "f1_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — ROC-AUC Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_rocauc_comparison(results_df: pd.DataFrame):
    df = results_df.sort_values("roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE.get(n, "#333333") for n in df["model"]]
    bars = ax.barh(df["model"], df["roc_auc"], color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, df["roc_auc"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("ROC-AUC (5-fold CV mean)", fontsize=11)
    ax.set_title("ROC-AUC Comparison — All Models", fontsize=13, fontweight="bold")
    ax.set_xlim(0.4, 1.05)
    ax.axvline(0.90, color="#E63946", linestyle="--", lw=1.5, alpha=0.7, label="AUC=0.90 target")
    ax.axvline(0.50, color="gray",   linestyle="--", lw=1,   alpha=0.5, label="Random baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "rocauc_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN OUT-OF-FOLD INFERENCE TO GET PROBS & PREDS
# ─────────────────────────────────────────────────────────────────────────────
def run_oof_inference(X, y, groups, device, folds, batch_size):
    """
    Run out-of-fold inference using saved model weights.
    Returns per-model dicts: {name: (oof_labels, oof_probs, oof_preds)}
    """
    sgkf   = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    model_dir = RESULTS_DIR / "models"
    criterion = nn.CrossEntropyLoss()

    oof_results = {}

    for model_name, model_class in MODEL_REGISTRY.items():
        print(f"  [OOF] {model_name} ...")
        all_labels, all_probs, all_preds = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(
                sgkf.split(X, y, groups), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            mean, std   = compute_fold_stats(X_tr)
            val_ds  = SolarFlareDataset(X_val, y_val, mean, std)
            val_ld  = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            model = model_class().to(device)
            weight_file = (model_dir /
                           f"{model_name.lower().replace(' ', '_')}_fold{fold_idx}.pt")
            if weight_file.exists():
                state = torch.load(weight_file, map_location=device)
                model.load_state_dict(state)
            else:
                print(f"    [WARN] Weight file not found: {weight_file} — using random init")

            _, probs, preds, labels = evaluate(model, val_ld, criterion, device)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

        oof_results[model_name] = (
            np.array(all_labels),
            np.array(all_probs),
            np.array(all_preds),
        )

    return oof_results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print(f"[INFO] Device: {device}")

    # Load dataset
    X, y, meta, version = load_dataset(args.data_dir)
    groups = meta["date"].values
    print(f"[INFO] X={X.shape} | y={y.shape} | groups={len(np.unique(groups))}")

    # ── Plot 1: Sample sequences ─────────────────────────────────────────────
    print("\n[PLOT 1] Sample Solar Flare Sequences")
    plot_sample_sequences(X, y)

    # ── Plot 2: Loss curves from histories ───────────────────────────────────
    hist_path = RESULTS_DIR / "loss_histories.json"
    if hist_path.exists():
        print("\n[PLOT 2] Loss Curves")
        with open(hist_path) as f:
            histories = json.load(f)
        plot_loss_curves(histories)
    else:
        print("[WARN] loss_histories.json not found — skipping loss curves.")
        print("       Run train.py first.")

    # ── Load benchmark results ───────────────────────────────────────────────
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if not csv_path.exists():
        print("[WARN] benchmark_results.csv not found. Run train.py first.")
        results_df = None
    else:
        results_df = pd.read_csv(csv_path)
        print(f"\n[INFO] Loaded benchmark_results.csv — {len(results_df)} models")

    # ── OOF inference for ROC/PR/CM plots ───────────────────────────────────
    print("\n[OOF] Running out-of-fold inference for ROC/PR/CM plots...")
    oof = run_oof_inference(X, y, groups, device, args.folds, args.batch_size)

    all_probs = {n: (v[0], v[1]) for n, v in oof.items()}
    all_preds = {n: (v[0], v[2]) for n, v in oof.items()}

    # ── Plot 3: ROC Curves ───────────────────────────────────────────────────
    print("\n[PLOT 3] ROC Curves")
    plot_roc_curves(all_probs)

    # ── Plot 4: PR Curves ────────────────────────────────────────────────────
    print("\n[PLOT 4] Precision-Recall Curves")
    plot_pr_curves(all_probs)

    # ── Plot 5: Confusion Matrices ───────────────────────────────────────────
    print("\n[PLOT 5] Confusion Matrices")
    plot_confusion_matrices(all_preds)

    if results_df is not None:
        # ── Plot 6: F1 comparison ────────────────────────────────────────────
        print("\n[PLOT 6] F1 Comparison")
        plot_f1_comparison(results_df)

        # ── Plot 7: ROC-AUC comparison ───────────────────────────────────────
        print("\n[PLOT 7] ROC-AUC Comparison")
        plot_rocauc_comparison(results_df)

    print("\n[DONE] All plots saved to plots/")


if __name__ == "__main__":
    main()
