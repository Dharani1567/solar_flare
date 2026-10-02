#!/usr/bin/env python3
"""
Benchmark Rebuilder for dataset_forecast_v2 (PRADAN Telemetry)

MODELS:
1. Logistic Regression
2. Random Forest
3. 1D CNN
4. ResNet1D

EVALUATION:
- StratifiedGroupKFold (5 splits, groups = flare_event_id)
- Zero flare-event leakage across folds
- Output saved to results/benchmark_metrics_v2.csv and export directory.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from models import Conv1DModel, ResNet1DModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPORT_DIR = PROJECT_ROOT / "dataset_forecast_v2"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_tabular_features(X: np.ndarray) -> np.ndarray:
    """Extract 36 summary statistical features per sample."""
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
    N, L, C = X.shape
    feats = []
    t = np.linspace(0, 1, L)
    
    for i in range(N):
        row = []
        for c in range(C):
            sig = X[i, :, c]
            mean_val = float(np.mean(sig))
            std_val = float(np.std(sig))
            min_val = float(np.min(sig))
            max_val = float(np.max(sig))
            p25 = float(np.percentile(sig, 25))
            p75 = float(np.percentile(sig, 75))
            median_val = float(np.median(sig))
            
            slope = float(np.polyfit(t, sig, 1)[0]) if len(sig) > 1 else 0.0
            snr = float(max_val / (std_val + 1e-6))
            
            row.extend([mean_val, std_val, min_val, max_val, median_val, p25, p75, slope, snr])
        feats.append(row)
        
    res = np.array(feats, dtype=np.float32)
    return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

def train_tabular_model(model_name: str, X_tab: np.ndarray, y: np.ndarray, groups: np.ndarray):
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tab, y, groups)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_tab[train_idx])
        X_val = scaler.transform(X_tab[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]
        
        if model_name == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        elif model_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds
        
    return oof_probs, oof_preds

def train_pytorch_model(model_name: str, model_cls, X_seq: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 15):
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_seq, y, groups)):
        mean_tr = np.mean(X_seq[train_idx], axis=(0, 1), keepdims=True)
        std_tr = np.std(X_seq[train_idx], axis=(0, 1), keepdims=True) + 1e-6
        
        X_tr_norm = (X_seq[train_idx] - mean_tr) / std_tr
        X_val_norm = (X_seq[val_idx] - mean_tr) / std_tr
        
        X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32).transpose(1, 2)
        y_tr_t = torch.tensor(y[train_idx], dtype=torch.long)
        
        X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).transpose(1, 2)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.long)
        
        train_ds = TensorDataset(X_tr_t, y_tr_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        model = model_cls(in_channels=4, num_classes=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for ep in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs_list.extend(probs)
                
        val_probs = np.array(val_probs_list)
        val_preds = (val_probs >= 0.5).astype(int)
        
        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds
        
    return oof_probs, oof_preds

def run_benchmark():
    print("=" * 75)
    print(" REBUILDING BENCHMARK FOR dataset_forecast_v2")
    print("=" * 75)
    
    X = np.load(DATA_DIR / "X_forecast_v2.npy")
    y = np.load(DATA_DIR / "y_forecast_v2.npy")
    df_meta = pd.read_csv(DATA_DIR / "forecast_metadata_v2.csv")
    groups = df_meta["flare_event_id"].values
    
    X_tab = extract_tabular_features(X)
    
    models = [
        ("Logistic Regression", "tabular"),
        ("Random Forest", "tabular"),
        ("1D CNN", Conv1DModel),
        ("ResNet1D", ResNet1DModel)
    ]
    
    results = []
    
    for name, m_type in models:
        print(f"\n[EVAL] Running {name}...")
        t0 = time.time()
        if m_type == "tabular":
            probs, preds = train_tabular_model(name, X_tab, y, groups)
        else:
            probs, preds = train_pytorch_model(name, m_type, X, y, groups, epochs=15)
        dt = time.time() - t0
        
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs)
        pr_auc = average_precision_score(y, probs)
        
        results.append({
            "model_name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "training_time_sec": round(dt, 2)
        })
        print(f"   --> F1: {f1:.4f} | ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_DIR / "benchmark_metrics_v2.csv", index=False)
    df_res.to_csv(EXPORT_DIR / "benchmark_metrics_v2.csv", index=False)
    
    print("\n" + "=" * 75)
    print(" BENCHMARK REBUILD COMPLETED (benchmark_metrics_v2.csv)")
    print("=" * 75)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
