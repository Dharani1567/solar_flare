#!/usr/bin/env python3
"""
Master Independent Scientific Validation & Kaggle Package Generation Pipeline
Executes Tasks 1-7 for dataset_forecast_v1.
"""

from __future__ import annotations

import os
import sys
import json
import time
import glob
import zipfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
META_DIR = PROJECT_ROOT / "data" / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_DIR / "X_forecast_v1.npy"
Y_PATH = DATA_DIR / "y_forecast_v1.npy"
META_PATH = DATA_DIR / "forecast_metadata_v1.csv"

def run_pipeline():
    print("=" * 75)
    print(" 🔬 INDEPENDENT SCIENTIFIC VALIDATION PIPELINE (dataset_forecast_v1)")
    print("=" * 75)

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    df_meta = pd.read_csv(META_PATH)

    tot_samples = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # Assign flare_event_id
    evt_keys = []
    for idx, row in df_meta.iterrows():
        if row["label"] == 1:
            key = f"FLARE_{row['flare_start_time']}_{row['flare_peak_time']}_{row['goes_class']}"
        else:
            key = f"QUIET_{row['observation_date']}"
        evt_keys.append(key)

    df_meta["flare_event_key"] = evt_keys
    key_to_id = {k: i + 1 for i, k in enumerate(df_meta["flare_event_key"].unique())}
    df_meta["flare_event_id"] = df_meta["flare_event_key"].map(key_to_id)
    groups = df_meta["flare_event_id"].values

    # =========================================================================
    # TASK 1: DATASET AUDIT & VISUALIZATIONS
    # =========================================================================
    print("\n--- TASK 1: DATASET AUDIT & VISUALIZATIONS ---")
    
    pos_df = df_meta[df_meta["label"] == 1]
    
    # Verification checks
    in_window_peaks = [i for i in pos_idx if np.max(X[i]) > 1.0]
    invalid_ts = pos_df[pos_df["minutes_until_flare"] <= 0.0]
    
    # Duplicate samples check
    X_flat_all = X.reshape(tot_samples, -1)
    unique_rows = np.unique(X_flat_all, axis=0)
    num_duplicates = tot_samples - len(unique_rows)
    
    print(f"Total Samples              : {tot_samples}")
    print(f"Positive Samples (y=1)     : {n_pos} ({n_pos/tot_samples*100:.1f}%)")
    print(f"Negative Samples (y=0)     : {n_neg} ({n_neg/tot_samples*100:.1f}%)")
    print(f"Unique Flare Events        : {len(pos_df['flare_event_key'].unique())}")
    print(f"In-Window Flare Visibility : {len(in_window_peaks)} ({len(in_window_peaks)/n_pos*100:.1f}%)")
    print(f"Invalid Timestamps (<=0m)  : {len(invalid_ts)}")
    print(f"Duplicate Samples Count    : {num_duplicates}")

    # Visualizations 1-4
    # Fig 1: Class Distribution
    plt.figure(figsize=(6, 4))
    plt.bar(["Quiet Sun (y=0)", "Pre-Flare (y=1)"], [n_neg, n_pos], color=["#2b5c8f", "#d9534f"], width=0.5)
    plt.ylabel("Sample Count")
    plt.title("Class Distribution (dataset_forecast_v1)")
    for i, v in enumerate([n_neg, n_pos]):
        plt.text(i, v + 15, f"{v} ({v/tot_samples*100:.1f}%)", ha="center", fontweight="bold")
    plt.ylim(0, 1000)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Fig 2: Horizon Distribution
    plt.figure(figsize=(7, 4))
    h_counts = pos_df["forecasting_horizon"].value_counts().sort_index()
    plt.bar(h_counts.index, h_counts.values, color="#428bca", width=0.5)
    plt.xlabel("Forecasting Lead Time Horizon")
    plt.ylabel("Positive Sample Count")
    plt.title("Pre-Flare Horizon Distribution")
    for i, v in enumerate(h_counts.values):
        plt.text(i, v + 2, str(v), ha="center", fontweight="bold")
    plt.ylim(0, 85)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "horizon_distribution.png", dpi=150)
    plt.close()

    # Fig 3: Samples per Flare Event
    evt_counts = pos_df.groupby("flare_event_key").size()
    plt.figure(figsize=(6, 4))
    plt.hist(evt_counts.values, bins=[1, 2, 3, 4, 5], align="left", color="#5cb85c", rwidth=0.6)
    plt.xlabel("Pre-Flare Windows per Event")
    plt.ylabel("Number of Unique Flare Events")
    plt.title("Window Extraction Distribution per Event")
    plt.xticks([1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "samples_per_flare_event.png", dpi=150)
    plt.close()

    # Fig 4: Feature Distributions
    f_max_0 = np.max(X_flat_all[y==0], axis=1)
    f_max_1 = np.max(X_flat_all[y==1], axis=1)
    plt.figure(figsize=(7, 4))
    plt.hist(f_max_0, bins=25, alpha=0.6, label="Quiet Sun (y=0)", color="blue", density=True)
    plt.hist(f_max_1, bins=25, alpha=0.6, label="Pre-Flare (y=1)", color="red", density=True)
    plt.xlabel("Maximum Input Signal Value")
    plt.ylabel("Density")
    plt.title("Pre-Flare vs Quiet Sun Signal Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_distributions.png", dpi=150)
    plt.close()

    print("[TASK 1 DONE] Audit visualizations generated in results/figures/")

    # =========================================================================
    # TASK 2: BASELINE MODELS & FEATURE ENGINEERING
    # =========================================================================
    print("\n--- TASK 2: TABULAR BASELINE MODELS ---")

    # Feature Engineering (15 features per channel * 4 channels = 60 tabular features)
    feat_list = []
    for i in range(tot_samples):
        seq = X[i] # (3600, 4)
        sample_feats = []
        for ch in range(4):
            c_data = seq[:, ch]
            c_diff = np.diff(c_data)
            
            sample_feats.extend([
                np.mean(c_data), np.median(c_data), np.std(c_data), np.var(c_data),
                np.min(c_data), np.max(c_data),
                np.percentile(c_data, 5), np.percentile(c_data, 25),
                np.percentile(c_data, 75), np.percentile(c_data, 95),
                skew(c_data), kurtosis(c_data),
                np.max(np.abs(c_diff)), np.mean(c_diff), np.std(c_diff)
            ])
        feat_list.append(sample_feats)
        
    X_tabular = np.array(feat_list, dtype=np.float32)
    print(f"Engineered Tabular Features Matrix Shape: {X_tabular.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(X, y, groups=groups))

    tabular_models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1, random_seed=42, verbose=0)
    }

    tabular_benchmark_rows = []

    for name, clf in tabular_models.items():
        print(f"Evaluating Tabular Model: {name}...")
        t0 = time.time()
        fold_m = []
        for tr, va in splits:
            m, s = np.mean(X_tabular[tr], axis=0), np.std(X_tabular[tr], axis=0) + 1e-8
            X_tr_sc = (X_tabular[tr] - m) / s
            X_va_sc = (X_tabular[va] - m) / s
            
            clf.fit(X_tr_sc, y[tr])
            p = clf.predict_proba(X_va_sc)[:, 1]
            pred = (p > 0.5).astype(int)
            
            fold_m.append({
                "acc": accuracy_score(y[va], pred), "prec": precision_score(y[va], pred, zero_division=0),
                "rec": recall_score(y[va], pred, zero_division=0), "f1": f1_score(y[va], pred, zero_division=0),
                "roc": roc_auc_score(y[va], p), "pr": average_precision_score(y[va], p)
            })
        t_el = time.time() - t0
        tabular_benchmark_rows.append({
            "model_name": name,
            "accuracy": round(np.mean([m['acc'] for m in fold_m]), 4),
            "precision": round(np.mean([m['prec'] for m in fold_m]), 4),
            "recall": round(np.mean([m['rec'] for m in fold_m]), 4),
            "f1_score": round(np.mean([m['f1'] for m in fold_m]), 4),
            "roc_auc": round(np.mean([m['roc'] for m in fold_m]), 4),
            "pr_auc": round(np.mean([m['pr'] for m in fold_m]), 4),
            "time_sec": round(t_el, 2)
        })

    df_tab_bench = pd.DataFrame(tabular_benchmark_rows).sort_values(by="f1_score", ascending=False)
    df_tab_bench.to_csv(RESULTS_DIR / "tabular_benchmarks.csv", index=False)
    print("\n=== TABULAR BASELINES BENCHMARK (results/tabular_benchmarks.csv) ===")
    print(df_tab_bench.to_string(index=False))

    # =========================================================================
    # TASK 3 & TASK 4: DEEP LEARNING & MULTI-SEED VALIDATION
    # =========================================================================
    print("\n--- TASK 3 & 4: DEEP LEARNING & MULTI-SEED VALIDATION ---")
    
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from models import MODEL_REGISTRY

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")

    seeds = [42, 123, 999]
    multiseed_records = []

    for name, cls in MODEL_REGISTRY.items():
        print(f"\n---> Evaluating Deep Model across 3 Seeds: {name.upper()}")
        seed_f1s, seed_rocs, seed_accs, seed_precs, seed_recs = [], [], [], [], []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            fold_m = []
            scaler_amp = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
            
            for fold, (tr, va) in enumerate(splits, 1):
                m_v = np.mean(X[tr], axis=(0, 1), keepdims=True)
                s_v = np.std(X[tr], axis=(0, 1), keepdims=True) + 1e-8
                
                X_tr = (X[tr] - m_v) / s_v
                X_va = (X[va] - m_v) / s_v
                
                ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y[tr]))
                ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y[va]))
                
                ldr_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
                ldr_va = DataLoader(ds_va, batch_size=32, shuffle=False)
                
                model = cls(in_channels=4, num_classes=2).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                best_val_f1 = -1.0
                ckpt_path = MODELS_DIR / f"{name}_seed{seed}_fold{fold}.pt"
                
                for epoch in range(1, 10):
                    model.train()
                    for xb_b, yb_b in ldr_tr:
                        xb_b, yb_b = xb_b.to(device), yb_b.to(device)
                        optimizer.zero_grad()
                        if scaler_amp and device.type == "cuda":
                            with torch.amp.autocast('cuda'):
                                out = model(xb_b)
                                loss = criterion(out, yb_b)
                            scaler_amp.scale(loss).backward()
                            scaler_amp.step(optimizer)
                            scaler_amp.update()
                        else:
                            out = model(xb_b)
                            loss = criterion(out, yb_b)
                            loss.backward()
                            optimizer.step()
                            
                    # Eval
                    model.eval()
                    p_l, t_l = [], []
                    with torch.no_grad():
                        for xb_b, yb_b in ldr_va:
                            p = torch.softmax(model(xb_b.to(device)), dim=1)[:, 1]
                            p_l.extend(p.cpu().numpy())
                            t_l.extend(yb_b.numpy())
                    f1_val = f1_score(np.array(t_l), (np.array(p_l)>0.5).astype(int), zero_division=0)
                    if f1_val > best_val_f1:
                        best_val_f1 = f1_val
                        torch.save(model.state_dict(), ckpt_path)
                        
                # Best evaluation
                if ckpt_path.exists():
                    model.load_state_dict(torch.load(ckpt_path))
                model.eval()
                p_l, t_l = [], []
                with torch.no_grad():
                    for xb_b, yb_b in ldr_va:
                        p = torch.softmax(model(xb_b.to(device)), dim=1)[:, 1]
                        p_l.extend(p.cpu().numpy())
                        t_l.extend(yb_b.numpy())
                        
                y_t, y_p = np.array(t_l), np.array(p_l)
                pred = (y_p > 0.5).astype(int)
                fold_m.append({
                    "acc": accuracy_score(y_t, pred), "prec": precision_score(y_t, pred, zero_division=0),
                    "rec": recall_score(y_t, pred, zero_division=0), "f1": f1_score(y_t, pred, zero_division=0),
                    "roc": roc_auc_score(y_t, y_p)
                })
                
            seed_accs.append(np.mean([m['acc'] for m in fold_m]))
            seed_precs.append(np.mean([m['prec'] for m in fold_m]))
            seed_recs.append(np.mean([m['rec'] for m in fold_m]))
            seed_f1s.append(np.mean([m['f1'] for m in fold_m]))
            seed_rocs.append(np.mean([m['roc'] for m in fold_m]))
            
        multiseed_records.append({
            "model_name": name.upper(),
            "accuracy_mean": round(np.mean(seed_accs), 4), "accuracy_std": round(np.std(seed_accs), 4),
            "precision_mean": round(np.mean(seed_precs), 4), "precision_std": round(np.std(seed_precs), 4),
            "recall_mean": round(np.mean(seed_recs), 4), "recall_std": round(np.std(seed_recs), 4),
            "f1_score_mean": round(np.mean(seed_f1s), 4), "f1_score_std": round(np.std(seed_f1s), 4),
            "roc_auc_mean": round(np.mean(seed_rocs), 4), "roc_auc_std": round(np.std(seed_rocs), 4),
            "f1_formatted": f"{np.mean(seed_f1s):.4f} +/- {np.std(seed_f1s):.4f}",
            "roc_formatted": f"{np.mean(seed_rocs):.4f} +/- {np.std(seed_rocs):.4f}"
        })

    df_multiseed = pd.DataFrame(multiseed_records).sort_values(by="f1_score_mean", ascending=False)
    df_multiseed.to_csv(RESULTS_DIR / "multiseed_summary.csv", index=False)
    print("\n=== MULTI-SEED SUMMARY (results/multiseed_summary.csv) ===")
    print(df_multiseed[["model_name", "f1_formatted", "roc_formatted"]].to_string(index=False))

    # =========================================================================
    # TASK 5: DISJOINT FORECAST HORIZON ANALYSIS
    # =========================================================================
    print("\n--- TASK 5: FORECAST HORIZON DEGRADATION ANALYSIS ---")
    
    horizon_intervals = {
        "Horizon A (0-1h)": (0.0, 60.0),
        "Horizon B (1-3h)": (60.0, 180.0),
        "Horizon C (3-6h)": (180.0, 360.0),
        "Horizon D (6-12h)": (360.0, 720.0)
    }

    horizon_analysis_rows = []
    ResNet1D_cls = MODEL_REGISTRY["resnet1d"]

    for h_name, (min_m, max_m) in horizon_intervals.items():
        pos_in_h = (y == 1) & (df_meta["minutes_until_flare"] > min_m) & (df_meta["minutes_until_flare"] <= max_m)
        h_mask = (y == 0) | pos_in_h
        
        X_h, y_h = X[h_mask], y[h_mask]
        meta_h = df_meta[h_mask].copy()
        
        n_pos_h = int(np.sum(y_h == 1))
        n_neg_h = int(np.sum(y_h == 0))
        tot_h = len(y_h)
        
        groups_h = meta_h["flare_event_id"].values
        sgkf_h = StratifiedGroupKFold(n_splits=5)
        splits_h = list(sgkf_h.split(X_h, y_h, groups=groups_h))
        
        rn_h = []
        for tr, va in splits_h:
            m_v, s_v = np.mean(X_h[tr], axis=(0,1), keepdims=True), np.std(X_h[tr], axis=(0,1), keepdims=True) + 1e-8
            ds_tr = TensorDataset(torch.from_numpy((X_h[tr]-m_v)/s_v), torch.from_numpy(y_h[tr]))
            ds_va = TensorDataset(torch.from_numpy((X_h[va]-m_v)/s_v), torch.from_numpy(y_h[va]))
            ldr_tr, ldr_va = DataLoader(ds_tr, batch_size=32, shuffle=True), DataLoader(ds_va, batch_size=32, shuffle=False)
            model = ResNet1D_cls(in_channels=4, num_classes=2).to(device)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            for epoch in range(1, 10):
                model.train()
                for xb_b, yb_b in ldr_tr:
                    xb_b, yb_b = xb_b.to(device), yb_b.to(device)
                    opt.zero_grad()
                    loss = crit(model(xb_b), yb_b)
                    loss.backward()
                    opt.step()
            model.eval()
            p_l, t_l = [], []
            with torch.no_grad():
                for xb_b, yb_b in ldr_va:
                    p = torch.softmax(model(xb_b.to(device)), dim=1)[:, 1]
                    p_l.extend(p.cpu().numpy())
                    t_l.extend(yb_b.numpy())
            y_t, y_p = np.array(t_l), np.array(p_l)
            rn_h.append({"f1": f1_score(y_t, (y_p>0.5).astype(int), zero_division=0), "roc": roc_auc_score(y_t, y_p)})
            
        horizon_analysis_rows.append({
            "horizon_name": h_name, "interval": f"{min_m/60:.0f}-{max_m/60:.0f}h",
            "total_samples": tot_h, "positive_count": n_pos_h, "negative_count": n_neg_h,
            "resnet_f1": round(np.mean([m['f1'] for m in rn_h]), 4),
            "resnet_roc": round(np.mean([m['roc'] for m in rn_h]), 4)
        })

    df_horiz = pd.DataFrame(horizon_analysis_rows)
    df_horiz.to_csv(RESULTS_DIR / "horizon_analysis.csv", index=False)
    print("\n=== DISJOINT HORIZON ANALYSIS (results/horizon_analysis.csv) ===")
    print(df_horiz.to_string(index=False))

    # Fig 5: Degradation Plot
    plt.figure(figsize=(7, 4))
    plt.plot(["0-1h", "1-3h", "3-6h", "6-12h"], df_horiz["resnet_f1"], "o-", label="F1 Score", linewidth=2.5, color="red")
    plt.plot(["0-1h", "1-3h", "3-6h", "6-12h"], df_horiz["resnet_roc"], "s-", label="ROC-AUC", linewidth=2.5, color="blue")
    plt.xlabel("Forecasting Lead Time Interval")
    plt.ylabel("Performance Metric Value")
    plt.title("ResNet1D Performance Degradation vs Lead Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.65, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "degradation_curves.png", dpi=150)
    plt.close()

    # =========================================================================
    # TASK 6: KAGGLE PACKAGES GENERATION
    # =========================================================================
    print("\n--- TASK 6: GENERATING KAGGLE ZIP PACKAGES ---")
    
    # 1. solar_flare_forecast_dataset.zip
    ds_zip_path = PROJECT_ROOT / "solar_flare_forecast_dataset.zip"
    with zipfile.ZipFile(ds_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(X_PATH, "X_forecast_v1.npy")
        z.write(Y_PATH, "y_forecast_v1.npy")
        z.write(META_PATH, "forecast_metadata_v1.csv")
        z.write(META_DIR / "forecast_statistics_v1.json", "forecast_statistics_v1.json")
        z.writestr("README.md", "# Solar Flare Forecast Dataset v1\nZero-leakage pre-flare forecasting dataset.")

    # 2. solar_flare_forecast_kaggle_package.zip
    pkg_zip_path = PROJECT_ROOT / "solar_flare_forecast_kaggle_package.zip"
    with zipfile.ZipFile(pkg_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(X_PATH, "dataset/X_forecast_v1.npy")
        z.write(Y_PATH, "dataset/y_forecast_v1.npy")
        z.write(META_PATH, "dataset/forecast_metadata_v1.csv")
        z.write(PROJECT_ROOT / "scripts" / "models.py", "models/models.py")
        
        # Add generated Kaggle notebooks
        nb_files = glob.glob("*.ipynb")
        for nb in nb_files:
            z.write(nb, f"notebooks/{nb}")
            
        z.writestr("requirements.txt", "torch>=2.0.0\nnumpy>=1.22.0\npandas>=1.5.0\nscikit-learn>=1.0.0\nxgboost>=1.6.0\nlightgbm>=3.3.0\ncatboost>=1.0.0\n")
        z.writestr("README.md", "# Kaggle Package Solar Flare Forecasting\nFull pipeline package.")

    print(f"[PACKAGE CREATED] Saved {ds_zip_path.name} ({ds_zip_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[PACKAGE CREATED] Saved {pkg_zip_path.name} ({pkg_zip_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\n[MASTER PIPELINE COMPLETE] All tasks 1-7 executed successfully!")

if __name__ == "__main__":
    run_pipeline()
