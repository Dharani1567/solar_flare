#!/usr/bin/env python3
"""
Rigorous Forecasting Leakage Audit — Flare-Event Level Grouping (dataset_forecast_v1)
Checks whether multiple pre-flare windows from the same flare event leak across folds.
Constructs flare_event_id from (flare_start_time, flare_peak_time, goes_class) and re-evaluates models.
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

X_PATH = DATA_DIR / "X_forecast_v1.npy"
Y_PATH = DATA_DIR / "y_forecast_v1.npy"
META_PATH = DATA_DIR / "forecast_metadata_v1.csv"

def run_flare_event_leakage_audit():
    print("=" * 75)
    print(" RIGOROUS FLARE-EVENT LEAKAGE AUDIT (dataset_forecast_v1)")
    print("=" * 75)

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    df_meta = pd.read_csv(META_PATH)

    # 1. Create unique flare_event_id using flare_start_time, flare_peak_time, goes_class
    evt_keys = []
    for idx, row in df_meta.iterrows():
        if row["label"] == 1:
            key = f"FLARE_{row['flare_start_time']}_{row['flare_peak_time']}_{row['goes_class']}"
        else:
            key = f"QUIET_{row['observation_date']}"
        evt_keys.append(key)

    df_meta["flare_event_key"] = evt_keys
    
    unique_event_keys = df_meta["flare_event_key"].unique()
    pos_event_keys = df_meta[df_meta["label"] == 1]["flare_event_key"].unique()
    neg_event_keys = df_meta[df_meta["label"] == 0]["flare_event_key"].unique()

    # Map to integer IDs
    key_to_id = {k: i + 1 for i, k in enumerate(unique_event_keys)}
    df_meta["flare_event_id"] = df_meta["flare_event_key"].map(key_to_id)
    groups_event = df_meta["flare_event_id"].values
    groups_date = df_meta["observation_date"].values

    print(f"\nTotal Sequence Samples : {len(df_meta)}")
    print(f"Total Unique Groups     : {len(unique_event_keys)}")
    print(f"Unique Positive Events  : {len(pos_event_keys)}")
    print(f"Unique Negative Events  : {len(neg_event_keys)}")

    # Samples per flare event statistics
    pos_df = df_meta[df_meta["label"] == 1]
    samples_per_pos_event = pos_df.groupby("flare_event_key").size()
    print("\nSamples per Positive Flare Event:")
    print(f"  - Mean samples per event : {samples_per_pos_event.mean():.2f}")
    print(f"  - Min samples per event  : {samples_per_pos_event.min()}")
    print(f"  - Max samples per event  : {samples_per_pos_event.max()}")

    # 2. Check overlap under Observation Date Grouping vs Flare Event Grouping
    sgkf_date = StratifiedGroupKFold(n_splits=5)
    splits_date = list(sgkf_date.split(X, y, groups=groups_date))

    overlap_count_date = 0
    for f, (tr, va) in enumerate(splits_date, 1):
        tr_evts = set(df_meta.loc[tr, "flare_event_key"])
        va_evts = set(df_meta.loc[va, "flare_event_key"])
        overlap = tr_evts.intersection(va_evts)
        overlap_count_date += len(overlap)

    sgkf_event = StratifiedGroupKFold(n_splits=5)
    splits_event = list(sgkf_event.split(X, y, groups=groups_event))

    overlap_count_event = 0
    for f, (tr, va) in enumerate(splits_event, 1):
        tr_evts = set(df_meta.loc[tr, "flare_event_key"])
        va_evts = set(df_meta.loc[va, "flare_event_key"])
        overlap = tr_evts.intersection(va_evts)
        overlap_count_event += len(overlap)

    print(f"\n--- Train/Validation Flare Event Overlap Count ---")
    print(f"  - Under Observation Date Grouping : {overlap_count_date} overlapping events (Target: 0)")
    print(f"  - Under Flare Event Grouping      : {overlap_count_event} overlapping events (Target: 0)")

    # 3. Retrain and Evaluate Benchmark Models under Flare Event Grouping
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from models import MODEL_REGISTRY
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_flat = X.reshape(len(y), -1)
    X_feats = np.column_stack([
        np.max(X_flat, axis=1), np.mean(X_flat, axis=1),
        np.std(X_flat, axis=1), np.percentile(X_flat, 95, axis=1)
    ])

    def benchmark_grouping(splits_list):
        # Logistic Regression
        lr_metrics = []
        for tr, va in splits_list:
            m, s = np.mean(X_feats[tr], axis=0), np.std(X_feats[tr], axis=0) + 1e-8
            clf = LogisticRegression().fit((X_feats[tr] - m)/s, y[tr])
            p = clf.predict_proba((X_feats[va] - m)/s)[:, 1]
            pred = (p > 0.5).astype(int)
            lr_metrics.append({"f1": f1_score(y[va], pred, zero_division=0), "roc": roc_auc_score(y[va], p)})

        # Random Forest
        rf_metrics = []
        for tr, va in splits_list:
            clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_feats[tr], y[tr])
            p = clf.predict_proba(X_feats[va])[:, 1]
            pred = (p > 0.5).astype(int)
            rf_metrics.append({"f1": f1_score(y[va], pred, zero_division=0), "roc": roc_auc_score(y[va], p)})

        # ResNet1D
        ResNet1D = MODEL_REGISTRY["resnet1d"]
        resnet_metrics = []
        scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

        for tr, va in splits_list:
            m_v, s_v = np.mean(X[tr], axis=(0,1), keepdims=True), np.std(X[tr], axis=(0,1), keepdims=True) + 1e-8
            ds_tr = TensorDataset(torch.from_numpy((X[tr]-m_v)/s_v), torch.from_numpy(y[tr]))
            ds_va = TensorDataset(torch.from_numpy((X[va]-m_v)/s_v), torch.from_numpy(y[va]))
            ldr_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
            ldr_va = DataLoader(ds_va, batch_size=32, shuffle=False)

            model = ResNet1D(in_channels=4, num_classes=2).to(device)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()

            for epoch in range(1, 10):
                model.train()
                for xb, yb in ldr_tr:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    if scaler and device.type == "cuda":
                        with torch.amp.autocast('cuda'):
                            out = model(xb)
                            loss = crit(out, yb)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        out = model(xb)
                        loss = crit(out, yb)
                        loss.backward()
                        opt.step()

            model.eval()
            p_list, t_list = [], []
            with torch.no_grad():
                for xb, yb in ldr_va:
                    p = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
                    p_list.extend(p.cpu().numpy())
                    t_list.extend(yb.numpy())

            y_t, y_p = np.array(t_list), np.array(p_list)
            resnet_metrics.append({"f1": f1_score(y_t, (y_p>0.5).astype(int), zero_division=0), "roc": roc_auc_score(y_t, y_p)})

        return {
            "lr_f1": float(np.mean([m['f1'] for m in lr_metrics])),
            "lr_roc": float(np.mean([m['roc'] for m in lr_metrics])),
            "rf_f1": float(np.mean([m['f1'] for m in rf_metrics])),
            "rf_roc": float(np.mean([m['roc'] for m in rf_metrics])),
            "resnet_f1": float(np.mean([m['f1'] for m in resnet_metrics])),
            "resnet_roc": float(np.mean([m['roc'] for m in resnet_metrics])),
        }

    print("\nEvaluating under Observation Date Grouping...")
    res_date = benchmark_grouping(splits_date)

    print("Evaluating under Flare Event Grouping...")
    res_event = benchmark_grouping(splits_event)

    print("\n" + "=" * 75)
    print(" COMPARISON: OBSERVATION DATE GROUPING vs FLARE EVENT GROUPING")
    print("=" * 75)
    print(f"Logistic Regression -> Date-Group F1: {res_date['lr_f1']:.4f} | Event-Group F1: {res_event['lr_f1']:.4f}")
    print(f"Logistic Regression -> Date-Group ROC: {res_date['lr_roc']:.4f} | Event-Group ROC: {res_event['lr_roc']:.4f}")
    print(f"Random Forest       -> Date-Group F1: {res_date['rf_f1']:.4f} | Event-Group F1: {res_event['rf_f1']:.4f}")
    print(f"Random Forest       -> Date-Group ROC: {res_date['rf_roc']:.4f} | Event-Group ROC: {res_event['rf_roc']:.4f}")
    print(f"ResNet1D (Top DL)   -> Date-Group F1: {res_date['resnet_f1']:.4f} | Event-Group F1: {res_event['resnet_f1']:.4f}")
    print(f"ResNet1D (Top DL)   -> Date-Group ROC: {res_date['resnet_roc']:.4f} | Event-Group ROC: {res_event['resnet_roc']:.4f}")

    audit_report_data = {
        "total_samples": len(df_meta),
        "total_event_groups": len(unique_event_keys),
        "positive_flare_events": len(pos_event_keys),
        "samples_per_pos_event_mean": float(samples_per_pos_event.mean()),
        "samples_per_pos_event_min": int(samples_per_pos_event.min()),
        "samples_per_pos_event_max": int(samples_per_pos_event.max()),
        "overlap_count_date_grouping": overlap_count_date,
        "overlap_count_event_grouping": overlap_count_event,
        "date_grouping": res_date,
        "event_grouping": res_event,
        "publication_quality": bool(overlap_count_date == 0 and overlap_count_event == 0 and res_event['resnet_roc'] > 0.95)
    }

    with open(SCRATCH_DIR / "flare_event_audit_results.json", "w") as f:
        json.dump(audit_report_data, f, indent=2)

    print("\nSaved flare event audit results to scratch/flare_event_audit_results.json")

if __name__ == "__main__":
    run_flare_event_leakage_audit()
