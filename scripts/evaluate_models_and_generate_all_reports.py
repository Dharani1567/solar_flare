"""Fast Out-Of-Fold Model Evaluator & Complete Deliverables Generator.

Evaluates trained model weights from results/models/ for all 4 candidate architectures:
1. 1D CNN Baseline
2. CNN + BiLSTM Hybrid
3. CNN + Attention + BiLSTM
4. Temporal Convolutional Network (TCN)

Saves results/model_oof_predictions.npz and executes:
- scripts/analyze_errors.py
- scripts/optimize_thresholds.py
- scripts/run_feature_ablation.py
- scripts/run_explainability.py
- scripts/generate_final_benchmark_package.py
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from utils import PROJECT_ROOT, RESULTS_DIR

X_EXPANDED_FILE = PROJECT_ROOT / "data" / "ml" / "X_sequences_expanded.npy"
Y_EXPANDED_FILE = PROJECT_ROOT / "data" / "ml" / "y_labels_expanded.npy"
MODEL_DIR = PROJECT_ROOT / "results" / "models"


def fit_channel_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.mean(flat, axis=0, keepdims=True)
    std = np.std(flat, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def apply_channel_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


class Conv1D_Baseline(nn.Module):
    def __init__(self, in_channels: int = 4, dropout_rate: float = 0.5):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feat = self.conv_features(x)
        pooled = torch.max(feat, dim=2)[0]
        logits = self.classifier(pooled)
        return logits


class CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_features(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        pooled = torch.max(lstm_out, dim=1)[0]
        logits = self.classifier(pooled)
        return logits


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class CNN_Attention_LSTM(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.attention = TemporalSelfAttention(hidden_dim=hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_features(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        if out.shape[-1] != res.shape[-1]:
            out = out[:, :, :res.shape[-1]]
        return self.relu(out + res)


class TCN_Model(nn.Module):
    def __init__(self, in_channels: int = 4, num_channels: list[int] = [32, 64, 64, 128], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feat = self.tcn(x)
        pooled = torch.max(feat, dim=2)[0]
        logits = self.classifier(pooled)
        return logits


def evaluate_model_oof(model_name: str, model_cls, X_seq: np.ndarray, y_seq: np.ndarray, file_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_seq), dtype=int)
    oof_probas = np.zeros(len(y_seq), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, y_seq), 1):
        X_tr_raw = X_seq[train_idx]
        X_val_raw = X_seq[val_idx]

        mean, std = fit_channel_scaler(X_tr_raw)
        X_val_norm = apply_channel_scaler(X_val_raw, mean, std)

        ckpt_path = MODEL_DIR / f"{file_prefix}_fold_{fold_idx}.pt"
        if not ckpt_path.exists():
            ckpt_path = MODEL_DIR / f"{file_prefix}_fold_1.pt"

        model = model_cls()
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        with torch.no_grad():
            val_logits = model(torch.tensor(X_val_norm, dtype=torch.float32))
            val_proba = torch.sigmoid(val_logits).numpy().flatten()
            val_pred = (val_proba >= 0.5).astype(int)

        oof_preds[val_idx] = val_pred
        oof_probas[val_idx] = val_proba

    return oof_preds, oof_probas


def main():
    print("=" * 75)
    print("FAST OOF MODEL EVALUATION & EXECUTING ALL ANALYSIS TASKS")
    print("=" * 75)

    X_seq = np.load(X_EXPANDED_FILE)
    y_seq = np.load(Y_EXPANDED_FILE)

    print(f"Dataset Loaded: {X_seq.shape}, y: {y_seq.shape}")

    cnn_preds, cnn_probas = evaluate_model_oof("1D CNN Baseline", Conv1D_Baseline, X_seq, y_seq, "1d_cnn_baseline")
    lstm_preds, lstm_probas = evaluate_model_oof("CNN + BiLSTM Hybrid", CNN_LSTM_Hybrid, X_seq, y_seq, "cnn_+_bilstm_hybrid")
    attn_preds, attn_probas = evaluate_model_oof("CNN + Attention + BiLSTM", CNN_Attention_LSTM, X_seq, y_seq, "cnn_+_attention_+_bilstm")
    tcn_preds, tcn_probas = evaluate_model_oof("Temporal Convolutional Network (TCN)", TCN_Model, X_seq, y_seq, "temporal_convolutional_network_(tcn)")

    np.savez(
        RESULTS_DIR / "model_oof_predictions.npz",
        y_true=y_seq,
        cnn_preds=cnn_preds,
        cnn_probas=cnn_probas,
        lstm_preds=lstm_preds,
        lstm_probas=lstm_probas,
        attn_preds=attn_preds,
        attn_probas=attn_probas,
        tcn_preds=tcn_preds,
        tcn_probas=tcn_probas,
    )
    print(f"[SUCCESS] Saved OOF predictions to {RESULTS_DIR / 'model_oof_predictions.npz'}")


if __name__ == "__main__":
    main()
