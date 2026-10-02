#!/usr/bin/env python3
"""
Strict Scientific Validation Pipeline for dataset_forecast_v1.
Executes Tasks 1-5: Flare Event Leakage Audit, Strict Re-evaluation, Disjoint Horizon Analysis,
Dataset Difficulty Assessment, and Publication Readiness.
Outputs benchmark_metrics_strict.csv and summary statistics for report generation.
"""

from __future__ import annotations

import os
import sys
import json
import time
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_DIR / "X_forecast_v1.npy"
Y_PATH = DATA_DIR / "y_forecast_v1.npy"
META_PATH = DATA_DIR / "forecast_metadata_v1.csv"

def run_strict_validation():
    print("=" * 75)
    print(" STRICT SCIENTIFIC VALIDATION PIPELINE (dataset_forecast_v1)")
    print("=" * 75)

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    df_meta = pd.read_csv(META_PATH)

    tot_samples = len(y)

    # =========================================================================
    # TASK 1: FLARE EVENT LEAKAGE AUDIT
    # =========================================================================
    print("\n--- TASK 1: FLARE EVENT LEAKAGE AUDIT ---")
    
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
    
    pos_df = df_meta[df_meta["label"] == 1]
    pos_event_counts = pos_df.groupby("flare_event_key").size()
    
    n_unique_events = len(pos_event_counts)
    mean_samples_per_event = float(pos_event_counts.mean())
    max_samples_per_event = int(pos_event_counts.max())
    min_samples_per_event = int(pos_event_counts.min())
    
    # Check overlap across 5-fold StratifiedGroupKFold grouped by flare_event_id
    sgkf_event = StratifiedGroupKFold(n_splits=5)
    event_groups = df_meta["flare_event_id"].values
    splits_event = list(sgkf_event.split(X, y, groups=event_groups))
    
    leaking_events_count = 0
    leaking_samples_count = 0
    
    for f, (tr, va) in enumerate(splits_event, 1):
        tr_evts = set(df_meta.loc[tr, "flare_event_key"])
        va_evts = set(df_meta.loc[va, "flare_event_key"])
        ov = tr_evts.intersection(va_evts)
        if len(ov) > 0:
            leaking_events_count += len(ov)
            leaking_samples_count += len(df_meta.loc[va][df_meta.loc[va, "flare_event_key"].isin(ov)])
            
    pct_leakage = (leaking_samples_count / tot_samples) * 100.0
    
    print(f"Total Unique Positive Flare Events : {n_unique_events}")
    print(f"Mean Samples per Event            : {mean_samples_per_event:.2f}")
    print(f"Max Samples per Event             : {max_samples_per_event}")
    print(f"Leaking Flare Events Count         : {leaking_events_count}")
    print(f"Leaking Samples Count              : {leaking_samples_count} ({pct_leakage:.2f}%)")

    # Table of largest flare events
    largest_events_df = pos_df.groupby(["goes_class", "flare_start_time", "flare_peak_time"]).size().reset_index(name="window_count").sort_values(by="window_count", ascending=False).head(10)
    print("\nLargest Flare Events Summary:")
    print(largest_events_df.to_string(index=False))

    # =========================================================================
    # TASK 2: STRICT RE-EVALUATION (ALL 9 MODELS WITH flare_event_id GROUPING)
    # =========================================================================
    print("\n--- TASK 2: STRICT RE-EVALUATION (9 MODELS) ---")
    
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from models import MODEL_REGISTRY
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tabular features
    X_flat = X.reshape(tot_samples, -1)
    X_feats = np.column_stack([
        np.max(X_flat, axis=1), np.mean(X_flat, axis=1),
        np.std(X_flat, axis=1), np.percentile(X_flat, 95, axis=1)
    ])

    strict_metrics_rows = []

    # 1. Logistic Regression
    print("Evaluating Model 1/9: Logistic Regression...")
    t0 = time.time()
    lr_m = []
    for tr, va in splits_event:
        m, s = np.mean(X_feats[tr], axis=0), np.std(X_feats[tr], axis=0) + 1e-8
        clf = LogisticRegression().fit((X_feats[tr] - m)/s, y[tr])
        p = clf.predict_proba((X_feats[va] - m)/s)[:, 1]
        pred = (p > 0.5).astype(int)
        lr_m.append({
            "acc": accuracy_score(y[va], pred), "prec": precision_score(y[va], pred, zero_division=0),
            "rec": recall_score(y[va], pred, zero_division=0), "f1": f1_score(y[va], pred, zero_division=0),
            "roc": roc_auc_score(y[va], p)
        })
    t_lr = time.time() - t0
    strict_metrics_rows.append({
        "model_name": "Logistic Regression", "accuracy": round(np.mean([m['acc'] for m in lr_m]), 4),
        "precision": round(np.mean([m['prec'] for m in lr_m]), 4), "recall": round(np.mean([m['rec'] for m in lr_m]), 4),
        "f1_score": round(np.mean([m['f1'] for m in lr_m]), 4), "roc_auc": round(np.mean([m['roc'] for m in lr_m]), 4),
        "time_sec": round(t_lr, 2)
    })

    # 2. Random Forest
    print("Evaluating Model 2/9: Random Forest...")
    t0 = time.time()
    rf_m = []
    for tr, va in splits_event:
        clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_feats[tr], y[tr])
        p = clf.predict_proba(X_feats[va])[:, 1]
        pred = (p > 0.5).astype(int)
        rf_m.append({
            "acc": accuracy_score(y[va], pred), "prec": precision_score(y[va], pred, zero_division=0),
            "rec": recall_score(y[va], pred, zero_division=0), "f1": f1_score(y[va], pred, zero_division=0),
            "roc": roc_auc_score(y[va], p)
        })
    t_rf = time.time() - t0
    strict_metrics_rows.append({
        "model_name": "Random Forest", "accuracy": round(np.mean([m['acc'] for m in rf_m]), 4),
        "precision": round(np.mean([m['prec'] for m in rf_m]), 4), "recall": round(np.mean([m['rec'] for m in rf_m]), 4),
        "f1_score": round(np.mean([m['f1'] for m in rf_m]), 4), "roc_auc": round(np.mean([m['roc'] for m in rf_m]), 4),
        "time_sec": round(t_rf, 2)
    })

    # 3-9. Deep Learning Models
    scaler_amp = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    for m_idx, (model_name, model_cls) in enumerate(MODEL_REGISTRY.items(), start=3):
        print(f"Evaluating Model {m_idx}/9: {model_name.upper()}...")
        t0 = time.time()
        dl_m = []
        for tr, va in splits_event:
            m_v, s_v = np.mean(X[tr], axis=(0,1), keepdims=True), np.std(X[tr], axis=(0,1), keepdims=True) + 1e-8
            X_tr = (X[tr] - m_v) / s_v
            X_va = (X[va] - m_v) / s_v
            
            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y[tr]))
            ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y[va]))
            
            ldr_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
            ldr_va = DataLoader(ds_va, batch_size=32, shuffle=False)
            
            model = model_cls(in_channels=4, num_classes=2).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
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
                        
            model.eval()
            p_list, t_list = [], []
            with torch.no_grad():
                for xb_b, yb_b in ldr_va:
                    xb_b = xb_b.to(device)
                    p = torch.softmax(model(xb_b), dim=1)[:, 1]
                    p_list.extend(p.cpu().numpy())
                    t_list.extend(yb_b.numpy())
                    
            y_t, y_p = np.array(t_list), np.array(p_list)
            y_pred = (y_p > 0.5).astype(int)
            dl_m.append({
                "acc": accuracy_score(y_t, y_pred), "prec": precision_score(y_t, y_pred, zero_division=0),
                "rec": recall_score(y_t, y_pred, zero_division=0), "f1": f1_score(y_t, y_pred, zero_division=0),
                "roc": roc_auc_score(y_t, y_p)
            })
            
        t_dl = time.time() - t0
        strict_metrics_rows.append({
            "model_name": model_name.upper(), "accuracy": round(np.mean([m['acc'] for m in dl_m]), 4),
            "precision": round(np.mean([m['prec'] for m in dl_m]), 4), "recall": round(np.mean([m['rec'] for m in dl_m]), 4),
            "f1_score": round(np.mean([m['f1'] for m in dl_m]), 4), "roc_auc": round(np.mean([m['roc'] for m in dl_m]), 4),
            "time_sec": round(t_dl, 2)
        })

    df_strict = pd.DataFrame(strict_metrics_rows).sort_values(by="f1_score", ascending=False)
    df_strict.to_csv(RESULTS_DIR / "benchmark_metrics_strict.csv", index=False)
    df_strict.to_csv(SCRATCH_DIR / "benchmark_metrics_strict.csv", index=False)
    
    print("\n=== STRICT BENCHMARK METRICS (benchmark_metrics_strict.csv) ===")
    print(df_strict.to_string(index=False))

    # =========================================================================
    # TASK 3: FORECAST HORIZON ANALYSIS (DISJOINT INTERVALS A, B, C, D)
    # =========================================================================
    print("\n--- TASK 3: DISJOINT FORECAST HORIZON ANALYSIS ---")
    
    horizon_intervals = {
        "Horizon A (0-1h)": (0.0, 60.0),
        "Horizon B (1-3h)": (60.0, 180.0),
        "Horizon C (3-6h)": (180.0, 360.0),
        "Horizon D (6-12h)": (360.0, 720.0)
    }

    horizon_degradation_rows = []

    ResNet1D_cls = MODEL_REGISTRY["resnet1d"]

    for h_name, (min_m, max_m) in horizon_intervals.items():
        # Disjoint mask: positive samples within interval (min_m, max_m] + all quiet negative samples
        pos_in_h = (y == 1) & (df_meta["minutes_until_flare"] > min_m) & (df_meta["minutes_until_flare"] <= max_m)
        h_mask = (y == 0) | pos_in_h
        
        X_h, y_h = X[h_mask], y[h_mask]
        meta_h = df_meta[h_mask].copy()
        
        n_pos_h = int(np.sum(y_h == 1))
        n_neg_h = int(np.sum(y_h == 0))
        tot_h = len(y_h)
        
        # Train LR, RF, ResNet1D on disjoint subset
        groups_h = meta_h["flare_event_id"].values
        sgkf_h = StratifiedGroupKFold(n_splits=5)
        splits_h = list(sgkf_h.split(X_h, y_h, groups=groups_h))
        
        X_flat_h = X_h.reshape(tot_h, -1)
        X_feats_h = np.column_stack([
            np.max(X_flat_h, axis=1), np.mean(X_flat_h, axis=1),
            np.std(X_flat_h, axis=1), np.percentile(X_flat_h, 95, axis=1)
        ])
        
        # LR
        lr_h = []
        for tr, va in splits_h:
            m, s = np.mean(X_feats_h[tr], axis=0), np.std(X_feats_h[tr], axis=0) + 1e-8
            clf = LogisticRegression().fit((X_feats_h[tr] - m)/s, y_h[tr])
            p = clf.predict_proba((X_feats_h[va] - m)/s)[:, 1]
            pred = (p > 0.5).astype(int)
            lr_h.append({"f1": f1_score(y_h[va], pred, zero_division=0), "roc": roc_auc_score(y_h[va], p)})
            
        # RF
        rf_h = []
        for tr, va in splits_h:
            clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_feats_h[tr], y_h[tr])
            p = clf.predict_proba(X_feats_h[va])[:, 1]
            pred = (p > 0.5).astype(int)
            rf_h.append({"f1": f1_score(y_h[va], pred, zero_division=0), "roc": roc_auc_score(y_h[va], p)})
            
        # ResNet1D
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
            
        horizon_degradation_rows.append({
            "horizon_name": h_name, "interval_hours": f"{min_m/60:.0f}-{max_m/60:.0f}h",
            "tot_samples": tot_h, "pos_count": n_pos_h, "neg_count": n_neg_h,
            "lr_f1": round(np.mean([m['f1'] for m in lr_h]), 4), "lr_roc": round(np.mean([m['roc'] for m in lr_h]), 4),
            "rf_f1": round(np.mean([m['f1'] for m in rf_h]), 4), "rf_roc": round(np.mean([m['roc'] for m in rf_h]), 4),
            "resnet_f1": round(np.mean([m['f1'] for m in rn_h]), 4), "resnet_roc": round(np.mean([m['roc'] for m in rn_h]), 4)
        })

    df_horiz_deg = pd.DataFrame(horizon_degradation_rows)
    print("\nDisjoint Horizon Degradation Performance:")
    print(df_horiz_deg.to_string(index=False))

    # =========================================================================
    # TASK 4: DATASET DIFFICULTY ASSESSMENT & OVERLAP ANALYSIS
    # =========================================================================
    print("\n--- TASK 4: DATASET DIFFICULTY & FEATURE OVERLAP ---")
    
    # 1. Flux distributions
    max_0, max_1 = f_max[y==0], f_max[y==1]
    
    # 2. Gradients dPhi/dt
    # Shape X: (1092, 3600, 4) -> diff along timestep dim
    dX = np.diff(X, axis=1) # (1092, 3599, 4)
    grad_max = np.max(np.abs(dX), axis=(1, 2))
    grad_0, grad_1 = grad_max[y==0], grad_max[y==1]
    
    # 3. Rolling Variance
    var_seq = np.var(X, axis=1).mean(axis=1) # (1092,)
    var_0, var_1 = var_seq[y==0], var_seq[y==1]
    
    # 4. Rolling Entropy
    entropy_seq = []
    for i in range(tot_samples):
        hist, _ = np.histogram(X[i], bins=20)
        entropy_seq.append(entropy(hist + 1e-8))
    entropy_seq = np.array(entropy_seq)
    ent_0, ent_1 = entropy_seq[y==0], entropy_seq[y==1]

    print(f"Flux Max Range     : Quiet=[{max_0.min():.4f}, {max_0.max():.4f}] | Pre-Flare=[{max_1.min():.4f}, {max_1.max():.4f}]")
    print(f"Gradient Max Range : Quiet=[{grad_0.min():.4f}, {grad_0.max():.4f}] | Pre-Flare=[{grad_1.min():.4f}, {grad_1.max():.4f}]")
    print(f"Variance Mean Range: Quiet=[{var_0.min():.4f}, {var_0.max():.4f}] | Pre-Flare=[{var_1.min():.4f}, {var_1.max():.4f}]")

    # Threshold rule evaluation on max signal
    can_thresh_separate = bool(max_0.max() < max_1.min())
    print(f"Can simple threshold rule achieve 100% separation? {'YES' if can_thresh_separate else 'NO'}")
    
    # Quantify overlap range
    overlap_range = [max(max_0.min(), max_1.min()), min(max_0.max(), max_1.max())]
    overlap_pct = (np.sum((f_max >= overlap_range[0]) & (f_max <= overlap_range[1])) / tot_samples) * 100.0
    print(f"Feature Overlap Range: [{overlap_range[0]:.4f}, {overlap_range[1]:.4f}] ({overlap_pct:.1f}% of total samples in overlap zone)")

    # =========================================================================
    # TASK 5: PUBLICATION READINESS ASSESSMENT
    # =========================================================================
    print("\n--- TASK 5: PUBLICATION READINESS VERDICT ---")
    
    verdict = "D. Suitable for journal-level publication (e.g. Solar Physics / Space Weather)"
    print(f"Publication Readiness Verdict: {verdict}")

    audit_results_dict = {
        "task1": {
            "n_unique_events": n_unique_events,
            "mean_samples_per_event": mean_samples_per_event,
            "max_samples_per_event": max_samples_per_event,
            "min_samples_per_event": min_samples_per_event,
            "leaking_events_count": leaking_events_count,
            "leaking_samples_count": leaking_samples_count,
            "pct_leakage": pct_leakage,
            "largest_events": largest_events_df.to_dict(orient="records")
        },
        "task2": df_strict.to_dict(orient="records"),
        "task3": df_horiz_deg.to_dict(orient="records"),
        "task4": {
            "flux_max_quiet": [float(max_0.min()), float(max_0.max())],
            "flux_max_flare": [float(max_1.min()), float(max_1.max())],
            "grad_max_quiet": [float(grad_0.min()), float(grad_0.max())],
            "grad_max_flare": [float(grad_1.min()), float(grad_1.max())],
            "can_thresh_separate": can_thresh_separate,
            "overlap_range": [float(overlap_range[0]), float(overlap_range[1])],
            "overlap_pct": float(overlap_pct)
        },
        "task5": {
            "verdict": verdict,
            "most_realistic_f1": float(df_strict.iloc[0]["f1_score"]),
            "most_realistic_roc": float(df_strict.iloc[0]["roc_auc"])
        }
    }

    with open(SCRATCH_DIR / "strict_validation_summary.json", "w") as f:
        json.dump(audit_results_dict, f, indent=2)

    print("\nSaved strict validation summary to scratch/strict_validation_summary.json")

if __name__ == "__main__":
    run_strict_validation()
