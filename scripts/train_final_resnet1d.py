#!/usr/bin/env python3
"""
Final Training & Production Script for Solar Flare Forecasting (ResNet1D Model)
Trains the final ResNet1D model on dataset_forecast_v2, saves weights to final_resnet1d.pth,
and generates publication-quality figures & evaluation reports.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

from models import ResNet1DModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_export_final_resnet1d():
    print("=" * 75)
    print(" PROJECT COMPLETION: TRAINING FINAL ResNet1D MODEL ON dataset_forecast_v2")
    print("=" * 75)

    X_path = DATA_DIR / "X_forecast_v2.npy"
    y_path = DATA_DIR / "y_forecast_v2.npy"
    meta_path = DATA_DIR / "forecast_metadata_v2.csv"

    X = np.load(X_path)
    y = np.load(y_path)
    df_meta = pd.read_csv(meta_path)

    N, L, C = X.shape
    print(f"[DATA LOADED] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[DEVICE] Using: {device}")

    # Channel-wise Standardization across full dataset
    mean_all = np.mean(X, axis=(0, 1), keepdims=True)
    std_all = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    X_norm = (X - mean_all) / std_all

    # Convert to PyTorch tensors: (N, C, L)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).transpose(1, 2)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 80/20 train/val split for loss curve tracking
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = ResNet1DModel(in_channels=4, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 35
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n[TRAINING] Starting 35-epoch training run...")
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct_tr = 0
        total_tr = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = torch.argmax(out, dim=1)
            correct_tr += (preds == by).sum().item()
            total_tr += len(by)

        epoch_tr_loss = running_loss / total_tr
        epoch_tr_acc = correct_tr / total_tr

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)

                running_val_loss += loss.item() * len(by)
                preds = torch.argmax(out, dim=1)
                correct_val += (preds == by).sum().item()
                total_val += len(by)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(epoch_tr_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_tr_acc)
        history["val_acc"].append(epoch_val_acc)

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {epoch_tr_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {epoch_tr_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

    # Full Dataset Inference & Evaluation
    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for bx, _ in full_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    # Calculate final metrics
    acc = accuracy_score(y, all_preds)
    prec = precision_score(y, all_preds, zero_division=0)
    rec = recall_score(y, all_preds, zero_division=0)
    f1 = f1_score(y, all_preds, zero_division=0)
    auc = roc_auc_score(y, all_probs)
    pr_auc = average_precision_score(y, all_probs)

    cm = confusion_matrix(y, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 75)
    print(" FINAL ResNet1D MODEL EVALUATION METRICS")
    print("=" * 75)
    print(f" Accuracy:       {acc:.4f}")
    print(f" Precision:      {prec:.4f}")
    print(f" Recall:         {rec:.4f}")
    print(f" F1 Score:       {f1:.4f}")
    print(f" ROC-AUC:        {auc:.4f}")
    print(f" PR-AUC:         {pr_auc:.4f}")
    print(f" Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Save Model Weights to final_resnet1d.pth
    weights_root = PROJECT_ROOT / "final_resnet1d.pth"
    weights_res = RESULTS_DIR / "final_resnet1d.pth"
    weights_ckpt = CHECKPOINT_DIR / "final_resnet1d.pth"

    torch.save(model.state_dict(), weights_root)
    torch.save(model.state_dict(), weights_res)
    torch.save(model.state_dict(), weights_ckpt)
    print(f"\n[MODEL SAVED] Model weights exported to {weights_root}")

    # -----------------------------------------------------------------
    # PUBLICATION-QUALITY FIGURES (300 DPI)
    # -----------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Confusion Matrix
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'},
                xticklabels=['Quiet (Negative)', 'Flare (Positive)'],
                yticklabels=['Quiet (Negative)', 'Flare (Positive)'])
    plt.title('Final ResNet1D Model — Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=10, fontweight='bold')
    plt.ylabel('True Ground Truth Label', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "confusion_matrix.png", dpi=300)
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, all_probs)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ResNet1D (ROC-AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=10, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "roc_curve.png", dpi=300)
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y, all_probs)
    plt.figure(figsize=(7, 5.5))
    plt.plot(recall_vals, precision_vals, color='#2ca02c', lw=2.5, label=f'ResNet1D (PR-AUC = {pr_auc:.4f})')
    plt.fill_between(recall_vals, precision_vals, alpha=0.15, color='#2ca02c')
    plt.xlabel('Recall (Sensitivity)', fontsize=10, fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=10, fontweight='bold')
    plt.title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='lower left', fontsize=10, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "precision_recall_curve.png", dpi=300)
    plt.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()

    # 4. Training Loss Curve
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(epochs_range, history["train_loss"], color='#d62728', lw=2.0, marker='o', ms=4, label='Training Loss')
    plt.plot(epochs_range, history["val_loss"], color='#ff7f0e', lw=2.0, marker='s', ms=4, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=10, fontweight='bold')
    plt.ylabel('Cross-Entropy Loss', fontsize=10, fontweight='bold')
    plt.title('ResNet1D Training & Validation Loss Trajectory', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='upper right', fontsize=10, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "training_loss_curve.png", dpi=300)
    plt.savefig(RESULTS_DIR / "training_loss_curve.png", dpi=300)
    plt.close()

    print("[SUCCESS] All 4 publication-quality figures saved successfully!")

    # Export final_results_report.md data structure
    report_dict = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    
    with open(RESULTS_DIR / "final_metrics_summary.json", "w") as f:
        json.dump(report_dict, f, indent=4)

if __name__ == "__main__":
    train_and_export_final_resnet1d()
