#!/usr/bin/env python3
"""
Generator for Kaggle GPU Notebooks:
1. kaggle_train_all_models.ipynb — Stratified 5-Fold training for 7 deep learning architectures with mixed precision on Kaggle T4/P100 GPUs.
2. kaggle_inference.ipynb — Real-time sample inference and metric visualization notebook.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_notebook_cell(source_lines, cell_type="code"):
    return {
        "cell_type": cell_type,
        "execution_count": None if cell_type == "code" else None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def generate_train_notebook():
    nb = {
        "cells": [
            create_notebook_cell([
                "# ☀️ Solar Flare Forecasting — 7-Architecture GPU Benchmark Training",
                "This notebook trains 7 deep learning architectures using **Stratified 5-Fold Cross-Validation** with **Mixed Precision (`torch.cuda.amp`)** on Kaggle GPU."
            ], cell_type="markdown"),

            create_notebook_cell([
                "# Step 1: Environment & GPU Inspection",
                "import torch",
                "import torch.nn as nn",
                "import numpy as np",
                "import pandas as pd",
                "from pathlib import Path",
                "",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
                "print(f'PyTorch Version: {torch.__version__}')",
                "print(f'CUDA Available: {torch.cuda.is_available()}')",
                "if torch.cuda.is_available():",
                "    print(f'GPU Device Name: {torch.cuda.get_device_name(0)}')",
                "    print(f'GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')"
            ]),

            create_notebook_cell([
                "# Step 2: Load Dataset v5",
                "dataset_dir = Path('../input/solar-flare-dataset-v5')",
                "if not dataset_dir.exists():",
                "    dataset_dir = Path('.')",
                "",
                "X_path = dataset_dir / 'X_sequences_v5.npy'",
                "y_path = dataset_dir / 'y_labels_v5.npy'",
                "meta_path = dataset_dir / 'sequence_metadata_v5.csv'",
                "",
                "X = np.load(X_path).astype(np.float32)",
                "y = np.load(y_path).astype(np.int64)",
                "meta = pd.read_csv(meta_path)",
                "",
                "print(f'X Tensor Shape: {X.shape} (Samples, Timesteps, Channels)')",
                "print(f'y Label Shape : {y.shape}')",
                "print(f'Positive Class (M/X Flares): {np.sum(y == 1)} ({np.sum(y==1)/len(y)*100:.1f}%)')",
                "print(f'Negative Class (Quiet Sun) : {np.sum(y == 0)} ({np.sum(y==0)/len(y)*100:.1f}%)')"
            ]),

            create_notebook_cell([
                "# Step 3: Run 7-Model Stratified 5-Fold CV Training",
                "import os",
                "os.system('python train_all_models.py')",
                "print('Training complete! Benchmark metrics saved to results/benchmark_metrics.csv')"
            ]),

            create_notebook_cell([
                "# Step 4: Display Benchmark Results",
                "bench_df = pd.read_csv('results/benchmark_metrics.csv')",
                "print('=== FINAL MODEL COMPARISON BENCHMARK ===')",
                "display(bench_df)"
            ])
        ],
        "metadata": {
            "language_info": {"name": "python"},
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = PROJECT_ROOT / "kaggle_train_all_models.ipynb"
    out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Generated {out_path.name}")

def generate_inference_notebook():
    nb = {
        "cells": [
            create_notebook_cell([
                "# 🚀 Solar Flare Forecasting — Kaggle Inference & Verification Notebook",
                "Loads `best_model.pt` and runs prediction on test sequences."
            ], cell_type="markdown"),

            create_notebook_cell([
                "import torch",
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "from models import MODEL_REGISTRY",
                "",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
                "print(f'Inference Device: {device}')"
            ]),

            create_notebook_cell([
                "# Load Optimal Model Weights & Sample Sequence",
                "X = np.load('X_sequences_v5.npy')[:5]",
                "y = np.load('y_labels_v5.npy')[:5]",
                "",
                "model = MODEL_REGISTRY['1d_cnn']()",
                "model.load_state_dict(torch.load('results/models/best_model.pt', map_location=device))",
                "model.to(device)",
                "model.eval()",
                "",
                "with torch.no_grad():",
                "    inputs = torch.from_numpy(X).to(device)",
                "    outputs = model(inputs)",
                "    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()",
                "",
                "for i in range(len(probs)):",
                "    pred_cls = 'Major Flare (M/X)' if probs[i] > 0.5847 else 'Quiet Sun / Minor'",
                "    print(f'Sample #{i+1} -> True Label: {y[i]} | Predicted Probability: {probs[i]:.4f} | Prediction: {pred_cls}')"
            ])
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = PROJECT_ROOT / "kaggle_inference.ipynb"
    out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Generated {out_path.name}")

if __name__ == "__main__":
    generate_train_notebook()
    generate_inference_notebook()
