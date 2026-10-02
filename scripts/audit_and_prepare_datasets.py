#!/usr/bin/env python3
"""
Comprehensive Dataset Audit & Variant Generator for Aditya-L1 Solar Flare Forecasting.
Audits:
  1. dataset_forecast_v2 (main dataset)
  2. pure_telemetry_dataset (strict validation dataset)
  3. cleaned_dataset (deduplicated, non-constant, non-corrupted variant)
Generates complete audit report with stats and creates dataset_cleaned.
"""

from __future__ import annotations
import os
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
V2_DIR = PROJECT_ROOT / "dataset_forecast_v2"
PURE_DIR = PROJECT_ROOT / "dataset_pure_telemetry"
CLEAN_DIR = PROJECT_ROOT / "dataset_cleaned"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def audit_dataset_variant(name: str, X: np.ndarray, y: np.ndarray, df_meta: pd.DataFrame):
    N, T, C = X.shape
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    ratio = neg_count / pos_count if pos_count > 0 else 0
    
    # Missing values
    nan_count = int(np.isnan(X).sum() + np.isnan(y).sum())
    inf_count = int(np.isinf(X).sum())
    
    # Constant sequences
    stds = np.std(X, axis=1) # (N, C)
    flat_all_ch = np.all(stds < 1e-4, axis=1)
    flat_any_ch = np.any(stds < 1e-4, axis=1)
    
    # Duplicate sequences
    hashes = [hashlib.sha256(X[i].tobytes()).hexdigest() for i in range(N)]
    df_h = pd.DataFrame({"hash": hashes, "label": y})
    dups = df_h.duplicated(subset=["hash"], keep=False)
    dup_count = int(dups.sum())
    unique_dup_patterns = int(df_h[dups]["hash"].nunique())
    
    # Outliers / corrupted samples (saturation flag >= 99.0 on CdTe channels 0/1)
    max_cdte = np.max(X[:, :, :2], axis=(1, 2))
    corrupted = int(np.sum(max_cdte >= 99.0))
    
    # Flare events and observation days
    date_col = "observation_date" if "observation_date" in df_meta.columns else "date"
    obs_days = int(df_meta[date_col].nunique())
    
    if "flare_event_id" in df_meta.columns:
        flare_events = int(df_meta[df_meta["label"] == 1]["flare_event_id"].nunique())
        group_col = "flare_event_id"
    else:
        flare_events = pos_count
        group_col = date_col
        
    # GroupKFold leakage check
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = df_meta[group_col].values
    leakage_count = 0
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        tr_flares = set(df_meta.iloc[tr][df_meta.iloc[tr]["label"] == 1][group_col])
        va_flares = set(df_meta.iloc[va][df_meta.iloc[va]["label"] == 1][group_col])
        leakage_count += len(tr_flares.intersection(va_flares))
        
    return {
        "dataset_name": name,
        "total_samples": N,
        "timesteps": T,
        "channels": C,
        "positive_samples": pos_count,
        "negative_samples": neg_count,
        "imbalance_ratio": round(ratio, 2),
        "positive_percent": round(pos_count / N * 100, 2),
        "observation_days": obs_days,
        "unique_flare_events": flare_events,
        "nan_values": nan_count,
        "inf_values": inf_count,
        "all_flat_sequences": int(np.sum(flat_all_ch)),
        "any_flat_channels": int(np.sum(flat_any_ch)),
        "duplicate_samples": dup_count,
        "unique_duplicate_patterns": unique_dup_patterns,
        "corrupted_spikes": corrupted,
        "group_kfold_leakage": leakage_count
    }

def main():
    print("=" * 75)
    print(" 1. LOADING DATASET_FORECAST_V2 (MAIN DATASET)")
    print("=" * 75)
    
    X_v2 = np.load(V2_DIR / "X_forecast_v2.npy").astype(np.float32)
    y_v2 = np.load(V2_DIR / "y_forecast_v2.npy").astype(np.int64)
    meta_v2 = pd.read_csv(V2_DIR / "forecast_metadata_v2.csv")
    
    audit_v2 = audit_dataset_variant("dataset_forecast_v2 (Main)", X_v2, y_v2, meta_v2)
    print("v2 audit:", json.dumps(audit_v2, indent=2))
    
    print("\n" + "=" * 75)
    print(" 2. LOADING DATASET_PURE_TELEMETRY (STRICT VALIDATION)")
    print("=" * 75)
    
    X_pure = np.load(PURE_DIR / "X_pure.npy").astype(np.float32)
    y_pure = np.load(PURE_DIR / "y_pure.npy").astype(np.int64)
    meta_pure = pd.read_csv(PURE_DIR / "metadata_pure.csv")
    
    audit_pure = audit_dataset_variant("dataset_pure_telemetry (Strict)", X_pure, y_pure, meta_pure)
    print("pure audit:", json.dumps(audit_pure, indent=2))
    
    print("\n" + "=" * 75)
    print(" 3. CONSTRUCTING CLEANED_DATASET VARIANT")
    print("=" * 75)
    
    # Filter 1: Remove all-flat placeholder sequences (std < 1e-4 across all 4 channels)
    stds = np.std(X_v2, axis=1)
    is_flat = np.all(stds < 1e-4, axis=1)
    
    # Filter 2: Remove corrupted saturation spikes on CdTe
    max_cdte = np.max(X_v2[:, :, :2], axis=(1, 2))
    is_corrupted = max_cdte >= 99.0
    
    keep_mask = (~is_flat) & (~is_corrupted)
    keep_indices = np.where(keep_mask)[0]
    
    X_clean = X_v2[keep_indices]
    y_clean = y_v2[keep_indices]
    meta_clean = meta_v2.iloc[keep_indices].copy().reset_index(drop=True)
    meta_clean["sample_id"] = np.arange(len(meta_clean))
    
    # Check deduplication on clean dataset
    hashes_clean = [hashlib.sha256(X_clean[i].tobytes()).hexdigest() for i in range(len(X_clean))]
    assert len(hashes_clean) == len(set(hashes_clean)), "Duplicates found in cleaned dataset!"
    
    # Save cleaned dataset
    np.save(CLEAN_DIR / "X_cleaned.npy", X_clean)
    np.save(CLEAN_DIR / "y_cleaned.npy", y_clean)
    meta_clean.to_csv(CLEAN_DIR / "metadata_cleaned.csv", index=False)
    
    # Also save to data/ml/
    np.save(DATA_DIR / "X_cleaned.npy", X_clean)
    np.save(DATA_DIR / "y_cleaned.npy", y_clean)
    meta_clean.to_csv(DATA_DIR / "metadata_cleaned.csv", index=False)
    
    audit_clean = audit_dataset_variant("cleaned_dataset (Deduplicated)", X_clean, y_clean, meta_clean)
    print("cleaned audit:", json.dumps(audit_clean, indent=2))
    
    # Combine audit summary table
    df_audit = pd.DataFrame([audit_v2, audit_clean, audit_pure])
    audit_csv = PROJECT_ROOT / "results" / "dataset_variants_audit.csv"
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    df_audit.to_csv(audit_csv, index=False)
    print(f"\n[SAVED] {audit_csv}")
    print(df_audit.to_string(index=False))

if __name__ == "__main__":
    main()
