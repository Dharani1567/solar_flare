#!/usr/bin/env python3
"""
Full 10-Model Benchmark Training & Evaluation Suite on dataset_forecast_v2 (PRADAN Telemetry)

MODELS:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. 1D CNN
5. CNN + BiLSTM
6. CNN + Attention + BiLSTM
7. TCN
8. Transformer
9. InceptionTime
10. ResNet1D

EVALUATION:
- StratifiedGroupKFold (5 splits, groups = flare_event_id)
- Zero flare-event leakage across folds
- Scaler / Normalization fit strictly on training folds
- Metrics saved to results/benchmark_metrics_v2.csv
- Plot Confusion Matrices, ROC Curves, PR Curves, and v1 vs v2 Comparison
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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

from models import (
    Conv1DModel, CNNBiLSTMModel, CNNAttentionBiLSTMModel,
    TCNModel, TransformerModel, InceptionTimeModel, ResNet1DModel
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
META_DIR = PROJECT_ROOT / "data" / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPORT_DIR = PROJECT_ROOT / "dataset_forecast_v2"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_tabular_features(X: np.ndarray) -> np.ndarray:
    """Extract 36 summary statistical features per sample from raw (N, 3600, 4) time series."""
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
            
            # Linear slope
            slope = float(np.polyfit(t, sig, 1)[0]) if len(sig) > 1 else 0.0
            snr = float(max_val / (std_val + 1e-6))
            
            row.extend([mean_val, std_val, min_val, max_val, median_val, p25, p75, slope, snr])
        feats.append(row)
        
    res = np.array(feats, dtype=np.float32)
    return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

def train_tabular_model(model_name: str, X_tab: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Evaluate tabular model using 5-fold StratifiedGroupKFold."""
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
        elif model_name == "XGBoost":
            clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
            
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds
        
    return oof_probs, oof_preds

def train_pytorch_model(model_name: str, model_cls, X_seq: np.ndarray, y: np.ndarray, groups: np.ndarray, epochs: int = 15):
    """Evaluate PyTorch 1D sequence model using 5-fold StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_seq, y, groups)):
        # Normalize along channel dimension using training fold stats only
        mean_tr = np.mean(X_seq[train_idx], axis=(0, 1), keepdims=True)
        std_tr = np.std(X_seq[train_idx], axis=(0, 1), keepdims=True) + 1e-6
        
        X_tr_norm = (X_seq[train_idx] - mean_tr) / std_tr
        X_val_norm = (X_seq[val_idx] - mean_tr) / std_tr
        
        # Shape: (B, 4, 3600)
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

def run_10_model_benchmark():
    print("=" * 75)
    print(" RUNNING 10-MODEL BENCHMARK ON dataset_forecast_v2 (PRADAN TELEMETRY)")
    print("=" * 75)
    
    # Load dataset_forecast_v2
    X_path = DATA_DIR / "X_forecast_v2.npy"
    y_path = DATA_DIR / "y_forecast_v2.npy"
    meta_path = DATA_DIR / "forecast_metadata_v2.csv"
    
    X = np.load(X_path)
    y = np.load(y_path)
    df_meta = pd.read_csv(meta_path)
    groups = df_meta["flare_event_id"].values
    
    print(f"Dataset Loaded: X shape {X.shape}, y shape {y.shape}, Unique Flare Groups: {len(np.unique(groups))}")
    print(f"Device: {device}")
    
    # Extract summary features for tabular models
    print("[FEAT] Extracting 36 statistical features for tabular models...")
    X_tab = extract_tabular_features(X)
    
    models_dict = {
        "Logistic Regression": "tabular",
        "Random Forest": "tabular",
        "XGBoost": "tabular",
        "1D CNN": Conv1DModel,
        "CNN + BiLSTM": CNNBiLSTMModel,
        "CNN + Attention + BiLSTM": CNNAttentionBiLSTMModel,
        "TCN": TCNModel,
        "Transformer": TransformerModel,
        "InceptionTime": InceptionTimeModel,
        "ResNet1D": ResNet1DModel
    }
    
    benchmark_results = []
    oof_predictions_dict = {}
    
    plt.figure(figsize=(10, 8))
    
    for model_name, model_type in models_dict.items():
        print(f"\n[TRAIN] Training model: {model_name}...")
        start_t = time.time()
        
        if model_type == "tabular":
            probs, preds = train_tabular_model(model_name, X_tab, y, groups)
        else:
            probs, preds = train_pytorch_model(model_name, model_type, X, y, groups, epochs=15)
            
        elapsed = time.time() - start_t
        
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs)
        pr_auc = average_precision_score(y, probs)
        
        benchmark_results.append({
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "training_time_sec": round(elapsed, 2)
        })
        
        oof_predictions_dict[model_name] = {"probs": probs, "preds": preds}
        
        print(f"  --> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")

    df_metrics = pd.DataFrame(benchmark_results)
    metrics_path = RESULTS_DIR / "benchmark_metrics_v2.csv"
    df_metrics.to_csv(metrics_path, index=False)
    df_metrics.to_csv(EXPORT_DIR / "benchmark_metrics_v2.csv", index=False)
    
    print("\n" + "=" * 75)
    print(" BENCHMARK RESULTS SUMMARY (dataset_forecast_v2)")
    print("=" * 75)
    print(df_metrics.to_string(index=False))
    
    # -----------------------------------------------------------------
    # PLOT ROC CURVES
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 7))
    for name, res in oof_predictions_dict.items():
        fpr, tpr, _ = roc_curve(y, res["probs"])
        auc_val = roc_auc_score(y, res["probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — 10 Models on dataset_forecast_v2 (PRADAN Telemetry)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves_v2.png", dpi=300)
    plt.savefig(EXPORT_DIR / "roc_curves_v2.png", dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # PLOT PRECISION-RECALL CURVES
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 7))
    for name, res in oof_predictions_dict.items():
        precision, recall, _ = precision_recall_curve(y, res["probs"])
        pr_auc_val = average_precision_score(y, res["probs"])
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc_val:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — 10 Models on dataset_forecast_v2")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pr_curves_v2.png", dpi=300)
    plt.savefig(EXPORT_DIR / "pr_curves_v2.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
    # PLOT CONFUSION MATRICES
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, (name, res) in enumerate(oof_predictions_dict.items()):
        cm = confusion_matrix(y, res["preds"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(name, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices_v2.png", dpi=300)
    plt.savefig(EXPORT_DIR / "confusion_matrices_v2.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
    # COMPARISON: v1 (SYNTHETIC PRECURSOR) vs v2 (REAL TELEMETRY)
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    v1_scores = [0.8814, 0.8650, 0.8520, 0.8710, 0.8750, 0.8780, 0.8790, 0.8640, 0.8800, 0.8814] # baseline v1 F1s
    v2_scores = df_metrics["f1_score"].values
    
    x_indices = np.arange(len(df_metrics))
    width = 0.35
    
    plt.bar(x_indices - width/2, v1_scores, width, label='v1 (Synthetic Precursor)', color='#3498db')
    plt.bar(x_indices + width/2, v2_scores, width, label='v2 (Real Telemetry)', color='#2ecc71')
    
    plt.xlabel('Model Architecture')
    plt.ylabel('F1 Score')
    plt.title('Performance Comparison: Synthetic (v1) vs Real PRADAN Telemetry (v2)')
    plt.xticks(x_indices, df_metrics["model_name"], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "v1_vs_v2_comparison.png", dpi=300)
    plt.savefig(EXPORT_DIR / "v1_vs_v2_comparison.png", dpi=300)
    plt.close()
    
    print("\n[COMPLETE] Benchmark run completed and all plots generated!")

if __name__ == "__main__":
    run_10_model_benchmark()
