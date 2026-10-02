"""Full Project Audit Data Collector.
Gathers exact metrics, storage footprints, model param counts, and file inventories.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

root = Path(__file__).resolve().parent.parent

def get_dir_size(path):
    total = 0
    for p in Path(path).rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return total

print("==================================================")
print("PROJECT AUDIT METRICS COLLECTOR")
print("==================================================")

# 1. Dataset stats
X_path = root / "data" / "ml" / "X_sequences_expanded.npy"
y_path = root / "data" / "ml" / "y_labels_expanded.npy"
meta_path = root / "data" / "ml" / "sequence_metadata_expanded.csv"

if X_path.exists():
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Dataset X Shape: {X.shape}")
    print(f"Dataset y Shape: {y.shape}")
    print(f"Dataset dtype: {X.dtype}")
    print(f"Class Distribution: Minor(0)={np.sum(y==0)}, Major(1)={np.sum(y==1)}")
    print(f"Imbalance Ratio: {np.sum(y==0)/np.sum(y==1):.2f}:1")
    print(f"NaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}")
    print(f"Dataset Tensor Size: {X.nbytes / (1024**2):.2f} MB")

if meta_path.exists():
    df_meta = pd.read_csv(meta_path)
    print(f"Metadata rows: {len(df_meta)}")
    if 'obs_date' in df_meta.columns:
        dates = pd.to_datetime(df_meta['obs_date']).dt.date.unique()
        print(f"Unique Observation Days: {len(dates)}")
        print(f"Date Range: {min(dates)} to {max(dates)}")

# 2. File Inventories
py_files = list(root.rglob("*.py"))
nb_files = list(root.rglob("*.ipynb"))
csv_files = list(root.rglob("*.csv"))
fits_files = list(root.rglob("*.fits")) + list(root.rglob("*.fits.gz"))
pt_files = list(root.rglob("*.pt")) + list(root.rglob("*.pth"))
pkl_files = list(root.rglob("*.pkl"))
md_files = list(root.rglob("*.md"))

print("\n--- FILE INVENTORY COUNTS ---")
print(f"Python Files (.py): {len(py_files)}")
print(f"Jupyter Notebooks (.ipynb): {len(nb_files)}")
print(f"CSV Files (.csv): {len(csv_files)}")
print(f"FITS Files (.fits/.fits.gz): {len(fits_files)}")
print(f"Model Checkpoints (.pt/.pth): {len(pt_files)}")
print(f"Pickle Files (.pkl): {len(pkl_files)}")
print(f"Documentation Files (.md): {len(md_files)}")

# 3. Storage Footprints
total_size = get_dir_size(root)
data_size = get_dir_size(root / "data") if (root / "data").exists() else 0
models_size = get_dir_size(root / "results" / "models") if (root / "results" / "models").exists() else 0
results_size = get_dir_size(root / "results") if (root / "results").exists() else 0

print("\n--- STORAGE FOOTPRINTS ---")
print(f"Total Project Size: {total_size / (1024**3):.3f} GB ({total_size / (1024**2):.2f} MB)")
print(f"Data Directory Size: {data_size / (1024**2):.2f} MB")
print(f"Model Weights Directory Size: {models_size / (1024**2):.2f} MB")
print(f"Results Directory Size: {results_size / (1024**2):.2f} MB")

# 4. Model Architectures Parameter Count
import sys
scripts_dir = root / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from train_architecture_comparison import (
    Conv1D_Baseline,
    CNN_LSTM_Hybrid,
    CNN_Attention_LSTM,
    TCN_Model,
)

models_dict = {
    "1D CNN Baseline": Conv1D_Baseline(),
    "CNN + BiLSTM Hybrid": CNN_LSTM_Hybrid(),
    "CNN + Attention + BiLSTM": CNN_Attention_LSTM(),
    "Temporal Convolutional Network (TCN)": TCN_Model(),
}

print("\n--- MODEL PARAMETER COUNTS ---")
for name, model in models_dict.items():
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - {name}: Total Params = {total_params:,} ({total_params * 4 / (1024**2):.2f} MB float32)")

# 5. Benchmark Performance
benchmark_csv = root / "results" / "benchmark_metrics.csv"
if benchmark_csv.exists():
    df_b = pd.read_csv(benchmark_csv)
    print("\n--- BENCHMARK PERFORMANCE TABLE ---")
    print(df_b.to_string(index=False))
