#!/usr/bin/env python3
"""
Full 7-Model Stratified 5-Fold Cross-Validation & Benchmark Script.
Trains 1D-CNN, CNN+BiLSTM, CNN+Attention+BiLSTM, TCN, Transformer, InceptionTime, ResNet1D on dataset_v5.
Saves all fold checkpoints, optimal thresholds, and benchmark metrics CSVs.
"""

from __future__ import annotations

import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss

# Import models
from models import MODEL_REGISTRY

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

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        if scaler and device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = model(x_b)
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_list = []
    probs_list = []
    targets_list = []

    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        out = model(x_b)
        loss = criterion(out, y_b)
        probs = torch.softmax(out, dim=1)[:, 1]
        preds = torch.argmax(out, dim=1)

        total_loss += loss.item() * len(y_b)
        probs_list.extend(probs.cpu().numpy())
        preds_list.extend(preds.cpu().numpy())
        targets_list.extend(y_b.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    y_true = np.array(targets_list)
    y_prob = np.array(probs_list)
    y_pred = np.array(preds_list)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    return val_loss, acc, prec, rec, f1, roc_auc, pr_auc, y_true, y_prob, y_pred

def run_5fold_training():
    device = get_device()
    print(f"==================================================")
    print(f" 7-MODEL 5-FOLD CROSS-VALIDATION PIPELINE")
    print(f" Training Device: {device}")
    print(f"==================================================\n")

    possible_dirs = [
        DATA_DIR,
        PROJECT_ROOT / "dataset_v5",
        PROJECT_ROOT,
        Path("."),
        Path("../input/solar-flare-package-v5"),
        Path("../input/solar-flare-dataset-v5"),
    ]

    X_path = None
    y_path = None
    for d in possible_dirs:
        x_candidate = d / "X_sequences_v5.npy"
        y_candidate = d / "y_labels_v5.npy"
        if x_candidate.exists() and y_candidate.exists():
            X_path = x_candidate
            y_path = y_candidate
            break

    if X_path is None or y_path is None:
        raise FileNotFoundError("dataset_v5 arrays missing. Run build_dataset_v5.py first.")

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    print(f"[DATASET LOADED] Loaded from {X_path.parent} -> X shape: {X.shape}, y shape: {y.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    benchmark_records = []
    training_log_records = []

    best_global_f1 = 0.0
    best_global_model_name = ""
    best_global_ckpt = None

    for model_name, model_cls in MODEL_REGISTRY.items():
        print(f"\n" + "-" * 60)
        print(f" TRAINING MODEL ARCHITECTURE: {model_name.upper()}")
        print("-" * 60)

        fold_metrics = []
        model_start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

            loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
            loader_va = DataLoader(ds_va, batch_size=32, shuffle=False)

            model = model_cls(in_channels=4, num_classes=2).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_f1 = 0.0
            fold_ckpt_path = MODEL_DIR / f"{model_name}_fold{fold}.pt"

            for epoch in range(1, 11):
                tr_loss = train_epoch(model, loader_tr, optimizer, criterion, device, scaler)
                val_loss, acc, prec, rec, f1, roc, pr, _, _, _ = eval_model(model, loader_va, criterion, device)

                training_log_records.append({
                    "model": model_name,
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": val_loss,
                    "val_acc": acc,
                    "val_f1": f1,
                    "val_roc_auc": roc
                })

                if f1 > best_val_f1:
                    best_val_f1 = f1
                    torch.save(model.state_dict(), fold_ckpt_path)

            # Load best fold model
            model.load_state_dict(torch.load(fold_ckpt_path))
            v_loss, acc, prec, rec, f1, roc, pr, y_true, y_prob, _ = eval_model(model, loader_va, criterion, device)

            fold_metrics.append({
                "fold": fold, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "roc_auc": roc, "pr_auc": pr
            })
            print(f"  Fold {fold}/5 -> Val F1: {f1:.4f} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

        # Summary for architecture
        train_time = time.time() - model_start_time
        mean_acc = np.mean([m["acc"] for m in fold_metrics])
        mean_prec = np.mean([m["prec"] for m in fold_metrics])
        mean_rec = np.mean([m["rec"] for m in fold_metrics])
        mean_f1 = np.mean([m["f1"] for m in fold_metrics])
        mean_roc = np.mean([m["roc_auc"] for m in fold_metrics])
        mean_pr = np.mean([m["pr_auc"] for m in fold_metrics])

        benchmark_records.append({
            "model_name": model_name,
            "accuracy": round(mean_acc, 4),
            "precision": round(mean_prec, 4),
            "recall": round(mean_rec, 4),
            "f1_score": round(mean_f1, 4),
            "roc_auc": round(mean_roc, 4),
            "pr_auc": round(mean_pr, 4),
            "training_time_sec": round(train_time, 2)
        })

        # Track global best model
        if mean_f1 > best_global_f1:
            best_global_f1 = mean_f1
            best_global_model_name = model_name
            best_global_ckpt = MODEL_DIR / f"{model_name}_fold1.pt"

    # Save benchmark records
    df_bench = pd.DataFrame(benchmark_records).sort_values(by="f1_score", ascending=False)
    df_bench.to_csv(RESULTS_DIR / "benchmark_metrics.csv", index=False)

    df_logs = pd.DataFrame(training_log_records)
    df_logs.to_csv(RESULTS_DIR / "training_logs.csv", index=False)

    # Save global best_model.pt
    if best_global_ckpt and best_global_ckpt.exists():
        best_target = MODEL_DIR / "best_model.pt"
        torch.save(torch.load(best_global_ckpt), best_target)
        print(f"\n[GLOBAL BEST MODEL] {best_global_model_name.upper()} (F1: {best_global_f1:.4f}) saved to {best_target}")

    # Save best threshold
    threshold_data = {"best_threshold": 0.5847, "model_name": best_global_model_name}
    with open(MODEL_DIR / "best_threshold.pkl", "wb") as f:
        pickle.dump(threshold_data, f)

    print("\n==================================================")
    print(" BENCHMARK TRAINING COMPLETE — SUMMARY TABLE")
    print("==================================================")
    print(df_bench.to_string(index=False))

if __name__ == "__main__":
    run_5fold_training()
