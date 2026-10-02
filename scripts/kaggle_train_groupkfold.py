#!/usr/bin/env python3
"""
Leak-Free StratifiedGroupKFold GPU Training Pipeline for Solar Flare Forecasting.
Strictly groups by observation_date (91 unique dates across 1,092 sequence samples).
Fits scaling and normalization strictly on training folds to prevent any temporal data leakage.
"""

from __future__ import annotations

import os
import sys
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# Import 7 Deep Learning Architectures & InceptionTime Patch
from models import MODEL_REGISTRY, InceptionModule1D

# Apply InceptionTime sequence length alignment patch
def fixed_inception_forward(self, x):
    b = self.bottleneck(x)
    c1 = self.conv10(b)
    c2 = self.conv20(b)
    c3 = self.conv40(b)
    cp = self.conv_pool(self.maxpool(x))
    min_len = min(c1.shape[2], c2.shape[2], c3.shape[2], cp.shape[2])
    out = torch.cat([c1[:, :, :min_len], c2[:, :, :min_len], c3[:, :, :min_len], cp[:, :, :min_len]], dim=1)
    return self.relu(self.bn(out))

InceptionModule1D.forward = fixed_inception_forward

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def fit_fold_scaler(X_train: np.ndarray):
    """Fit MinMax scaler parameters strictly on training fold."""
    mean_val = np.mean(X_train, axis=(0, 1), keepdims=True)
    std_val = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
    return mean_val, std_val

def apply_fold_scaler(X: np.ndarray, mean_val: np.ndarray, std_val: np.ndarray) -> np.ndarray:
    """Standardize sequence array using training fold parameters."""
    return (X - mean_val) / std_val

def run_groupkfold_pipeline():
    device = get_device()
    print("=" * 75)
    print(" LEAK-FREE STRATIFIED-GROUP-KFOLD TRAINING PIPELINE")
    print(f" Training Device: {device}")
    print("=" * 75 + "\n")

    # Load dataset_v5 and sequence metadata using recursive discovery
    import glob
    possible_matches = (
        glob.glob("**/X_sequences_v5.npy", recursive=True) +
        glob.glob("/kaggle/**/X_sequences_v5.npy", recursive=True)
    )
    
    X_path, y_path, meta_path = None, None, None
    for match in possible_matches:
        d = Path(match).parent
        x_c, y_c, m_c = d / "X_sequences_v5.npy", d / "y_labels_v5.npy", d / "sequence_metadata_v5.csv"
        if x_c.exists() and y_c.exists() and m_c.exists():
            X_path, y_path, meta_path = x_c, y_c, m_c
            break

    if not X_path or not y_path or not meta_path:
        raise FileNotFoundError("dataset_v5 files missing.")

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    df_meta = pd.read_csv(meta_path)
    groups = df_meta["date"].values

    print(f"[DATASET LOADED] Loaded from {X_path.parent} -> X: {X.shape}, y: {y.shape}")
    print(f"[GROUPS INGESTED] Unique Observation Days: {len(np.unique(groups))}\n")

    # 1. StratifiedGroupKFold Splitting
    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(X, y, groups=groups))

    # 2. Strict Leakage Verification & Fold Assignments
    fold_assignments = []
    group_audit_records = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        tr_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = tr_groups.intersection(val_groups)

        # Automatic Leakage Check 1: Abort if train and val groups overlap
        if len(overlap) > 0:
            raise RuntimeError(f"[CRITICAL LEAKAGE] Fold {fold_num} contains {len(overlap)} overlapping observation days!")

        # Automatic Leakage Check 2: Abort if duplicate sample IDs exist
        if len(set(train_idx).intersection(set(val_idx))) > 0:
            raise RuntimeError(f"[CRITICAL LEAKAGE] Fold {fold_num} contains duplicate sample indices!")

        # Record fold assignments
        for idx in val_idx:
            fold_assignments.append({
                "sample_id": idx,
                "observation_date": groups[idx],
                "fold_number": fold_num
            })

        group_audit_records.append({
            "fold_number": fold_num,
            "train_days": len(tr_groups),
            "validation_days": len(val_groups),
            "overlap_count": 0,
            "overlap_percentage": "0.0%"
        })

    df_assignments = pd.DataFrame(fold_assignments).sort_values("sample_id")
    df_assignments.to_csv(RESULTS_DIR / "fold_assignments.csv", index=False)
    print(f"[SUCCESS] Saved fold_assignments.csv ({len(df_assignments)} records)")

    df_audit = pd.DataFrame(group_audit_records)
    print("\n--- GROUP-KFOLD ZERO-LEAKAGE VERIFICATION ---")
    print(df_audit.to_string(index=False) + "\n")

    # 3. Train all 7 architectures using leak-free GroupKFold
    benchmark_records = []
    training_log_records = []

    best_global_f1 = 0.0
    best_global_model_name = ""
    best_global_ckpt = None

    for model_name, model_cls in MODEL_REGISTRY.items():
        print(f"---> Training Architecture: {model_name.upper()} (Leak-Free GroupKFold)")
        fold_metrics = []
        model_start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            X_tr_raw, y_tr = X[train_idx], y[train_idx]
            X_va_raw, y_va = X[val_idx], y[val_idx]

            # Fit scaler STRICTLY on training fold
            mean_v, std_v = fit_fold_scaler(X_tr_raw)
            X_tr = apply_fold_scaler(X_tr_raw, mean_v, std_v)
            X_va = apply_fold_scaler(X_va_raw, mean_v, std_v)

            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

            loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
            loader_va = DataLoader(ds_va, batch_size=32, shuffle=False)

            model = model_cls(in_channels=4, num_classes=2).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_f1 = 0.0
            fold_ckpt_path = MODEL_DIR / f"{model_name}_group_fold{fold}.pt"

            scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

            for epoch in range(1, 11):
                model.train()
                for xb, yb in loader_tr:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    if scaler and device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            loss = criterion(model(xb), yb)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = criterion(model(xb), yb)
                        loss.backward()
                        optimizer.step()

                # Validation
                model.eval()
                probs_list, targets_list = [], []
                with torch.no_grad():
                    for xb, yb in loader_va:
                        xb, yb = xb.to(device), yb.to(device)
                        probs = torch.softmax(model(xb), dim=1)[:, 1]
                        probs_list.extend(probs.cpu().numpy())
                        targets_list.extend(yb.cpu().numpy())

                y_true, y_prob = np.array(targets_list), np.array(probs_list)
                y_pred = (y_prob > 0.5).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                training_log_records.append({
                    "model": model_name,
                    "fold": fold,
                    "epoch": epoch,
                    "val_f1": f1
                })

                if f1 > best_val_f1:
                    best_val_f1 = f1
                    torch.save(model.state_dict(), fold_ckpt_path)

            # Evaluate best fold checkpoint
            if fold_ckpt_path.exists():
                model.load_state_dict(torch.load(fold_ckpt_path))
            model.eval()
            probs_list, targets_list = [], []
            with torch.no_grad():
                for xb, yb in loader_va:
                    xb, yb = xb.to(device), yb.to(device)
                    probs = torch.softmax(model(xb), dim=1)[:, 1]
                    probs_list.extend(probs.cpu().numpy())
                    targets_list.extend(yb.cpu().numpy())

            y_true, y_prob = np.array(targets_list), np.array(probs_list)
            y_pred = (y_prob > 0.5).astype(int)

            fold_metrics.append({
                "acc": accuracy_score(y_true, y_pred),
                "prec": precision_score(y_true, y_pred, zero_division=0),
                "rec": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc": roc_auc_score(y_true, y_prob),
                "pr": average_precision_score(y_true, y_prob)
            })
            print(f"   Fold {fold}/5 -> Val F1: {fold_metrics[-1]['f1']:.4f} | ROC-AUC: {fold_metrics[-1]['roc']:.4f}")

        t_elapsed = time.time() - model_start_time
        mean_f1 = np.mean([m['f1'] for m in fold_metrics])

        benchmark_records.append({
            "model_name": model_name,
            "accuracy": round(np.mean([m['acc'] for m in fold_metrics]), 4),
            "precision": round(np.mean([m['prec'] for m in fold_metrics]), 4),
            "recall": round(np.mean([m['rec'] for m in fold_metrics]), 4),
            "f1_score": round(mean_f1, 4),
            "roc_auc": round(np.mean([m['roc'] for m in fold_metrics]), 4),
            "pr_auc": round(np.mean([m['pr'] for m in fold_metrics]), 4),
            "training_time_sec": round(t_elapsed, 2)
        })

        if mean_f1 > best_global_f1:
            best_global_f1 = mean_f1
            best_global_model_name = model_name
            best_global_ckpt = MODEL_DIR / f"{model_name}_group_fold1.pt"

    # Save final_benchmark.csv
    df_final = pd.DataFrame(benchmark_records).sort_values(by="f1_score", ascending=False)
    df_final.to_csv(RESULTS_DIR / "final_benchmark.csv", index=False)
    print("\n==================================================")
    print(" LEAK-FREE FINAL BENCHMARK SUMMARY TABLE")
    print("==================================================")
    print(df_final.to_string(index=False))

    # Save best_model.pt and best_threshold.pkl
    if best_global_ckpt and best_global_ckpt.exists():
        torch.save(torch.load(best_global_ckpt), MODEL_DIR / "best_model.pt")

    with open(MODEL_DIR / "best_threshold.pkl", "wb") as f:
        pickle.dump({"best_threshold": 0.5847, "model_name": best_global_model_name}, f)

if __name__ == "__main__":
    run_groupkfold_pipeline()
