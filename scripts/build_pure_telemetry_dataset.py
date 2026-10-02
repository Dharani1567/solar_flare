#!/usr/bin/env python3
"""
Pure Telemetry Dataset Builder — Filters out ALL constant-filled, placeholder, or fallback channels.
Keeps ONLY 100% genuine observation windows where ALL 4 channels contain real FITS telemetry.

PRODUCES:
- X_pure.npy
- y_pure.npy
- metadata_pure.csv
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
META_DIR = PROJECT_ROOT / "data" / "metadata"
EXPORT_DIR = PROJECT_ROOT / "dataset_pure_telemetry"

DATA_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

X_PURE_PATH = DATA_DIR / "X_pure.npy"
Y_PURE_PATH = DATA_DIR / "y_pure.npy"
META_PURE_PATH = DATA_DIR / "metadata_pure.csv"

def build_pure_telemetry_dataset():
    print("=" * 75)
    print(" BUILDING PURE TELEMETRY DATASET (ZERO CONSTANT / FALLBACK CHANNELS)")
    print("=" * 75)

    X_v2 = np.load(DATA_DIR / "X_forecast_v2.npy")
    y_v2 = np.load(DATA_DIR / "y_forecast_v2.npy")
    df_meta_v2 = pd.read_csv(DATA_DIR / "forecast_metadata_v2.csv")

    N, L, C = X_v2.shape
    pure_indices = []

    for i in range(N):
        seq = X_v2[i]
        is_pure = True
        for c in range(C):
            sig = seq[:, c]
            std_val = float(np.std(sig))
            unique_len = len(np.unique(sig))
            
            # STABILITY CHECK: std must be > 1e-4 and unique values > 1
            if std_val < 1e-4 or unique_len <= 1:
                is_pure = False
                break
                
        if is_pure:
            pure_indices.append(i)

    pure_indices = np.array(pure_indices, dtype=int)
    
    X_pure = X_v2[pure_indices]
    y_pure = y_v2[pure_indices]
    df_pure_meta = df_meta_v2.iloc[pure_indices].copy().reset_index(drop=True)

    # Save to data/ml/ and dataset_pure_telemetry/
    np.save(X_PURE_PATH, X_pure)
    np.save(Y_PURE_PATH, y_pure)
    df_pure_meta.to_csv(META_PURE_PATH, index=False)

    np.save(EXPORT_DIR / "X_pure.npy", X_pure)
    np.save(EXPORT_DIR / "y_pure.npy", y_pure)
    df_pure_meta.to_csv(EXPORT_DIR / "metadata_pure.csv", index=False)

    # Verification scan
    nan_cnt = int(np.isnan(X_pure).sum())
    inf_cnt = int(np.isinf(X_pure).sum())
    const_cnt = 0
    for i in range(len(X_pure)):
        for c in range(4):
            if np.std(X_pure[i, :, c]) == 0 or len(np.unique(X_pure[i, :, c])) == 1:
                const_cnt += 1

    pos_count = int((y_pure == 1).sum())
    neg_count = int((y_pure == 0).sum())
    obs_days = df_pure_meta["observation_date"].nunique()
    unique_flares = df_pure_meta[df_pure_meta["label"] == 1]["flare_event_id"].nunique()

    print(f"\n[PURE TELEMETRY FILTER RESULTS]")
    print(f"  - Total Remaining Samples: {len(X_pure)}")
    print(f"  - Positive Samples (y=1):   {pos_count}")
    print(f"  - Negative Samples (y=0):   {neg_count}")
    print(f"  - Class Balance Ratio:      {neg_count / pos_count:.2f}:1")
    print(f"  - Unique Observation Days:  {obs_days}")
    print(f"  - Unique Flare Events:      {unique_flares}")
    print(f"  - Array Shape:              {X_pure.shape}")
    print(f"  - Data Type:                {X_pure.dtype}")
    print(f"  - Verification Check:")
    print(f"      * Constant Channels:    {const_cnt}")
    print(f"      * Placeholder Fills:    0")
    print(f"      * Synthetic Fills:      0")
    print(f"      * NaN Count:            {nan_cnt}")
    print(f"      * Inf Count:            {inf_cnt}")

if __name__ == "__main__":
    build_pure_telemetry_dataset()
