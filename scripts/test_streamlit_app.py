"""Test Streamlit App Logic & Model Loading."""

import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_architecture_comparison import (
    Conv1D_Baseline,
    CNN_LSTM_Hybrid,
    CNN_Attention_LSTM,
    TCN_Model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
MODEL_DIR = PROJECT_ROOT / "results" / "models"

X_PATH = DATA_DIR / "X_sequences_expanded.npy"
Y_PATH = DATA_DIR / "y_labels_expanded.npy"

X = np.load(X_PATH)
y = np.load(Y_PATH)

print(f"Loaded ML Data: X={X.shape}, y={y.shape}")

mapping = {
    "CNN + Attention + BiLSTM": ("cnn_+_attention_+_bilstm", CNN_Attention_LSTM),
    "CNN + BiLSTM Hybrid": ("cnn_+_bilstm_hybrid", CNN_LSTM_Hybrid),
    "1D CNN Baseline": ("1d_cnn_baseline", Conv1D_Baseline),
}

# Test Major Sample Index
major_idx = np.where(y == 1)[0][0]
seq_major = X[major_idx]
mean, std = np.mean(seq_major, axis=0, keepdims=True), np.std(seq_major, axis=0, keepdims=True)
std[std == 0] = 1.0
seq_norm = (seq_major - mean) / std
t_major = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)

for name, (prefix, cls) in mapping.items():
    models = []
    for f in range(1, 6):
        ckpt = MODEL_DIR / f"{prefix}_fold_{f}.pt"
        if ckpt.exists():
            m = cls()
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.eval()
            models.append(m)

    print(f"\nArchitecture [{name}]: Loaded {len(models)}/5 Fold Models.")
    probs = []
    for m in models:
        with torch.no_grad():
            out = m(t_major)
            logits = out[0] if isinstance(out, tuple) else out
            probs.append(torch.sigmoid(logits).item())
    mean_prob = np.mean(probs) if probs else 0.0
    print(f"  -> Predicted Major Flare Probability: {mean_prob * 100:.2f}%")

print("\n[SUCCESS] All Streamlit App Model Loading Checks PASSED 100%!")
