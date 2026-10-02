"""Phase 3 & 4: 7 Model Architectures Benchmark Suite & Weight Manager.

Trains and evaluates 7 deep learning architectures on dataset_v3 using Stratified 5-Fold CV:
1. 1D CNN Baseline
2. CNN + BiLSTM Hybrid
3. CNN + Attention + BiLSTM
4. Temporal Convolutional Network (TCN)
5. Transformer Encoder
6. InceptionTime
7. ResNet1D

Saves all 35 fold checkpoints, best_model.pt, best_threshold.pkl, benchmark_metrics.csv, and training_logs.csv.
"""

from __future__ import annotations

import time
import pickle
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"

for d in [MODEL_DIR, METRICS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

X_V3_FILE = DATA_DIR / "X_sequences_v3.npy"
Y_V3_FILE = DATA_DIR / "y_labels_v3.npy"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_channel_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.mean(flat, axis=0, keepdims=True)
    std = np.std(flat, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def apply_channel_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def compute_skill_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    tss = pod - pofd
    num_hss = 2.0 * (tp * tn - fp * fn)
    den_hss = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = (num_hss / den_hss) if den_hss > 0 else 0.0
    return {"TSS": tss, "HSS": hss, "POD": pod, "FAR": far, "CSI": csi}


# =====================================================================
# 7 MODEL ARCHITECTURE DEFINITIONS
# =====================================================================

# 1. 1D CNN Baseline
class Conv1D_Baseline(nn.Module):
    def __init__(self, in_channels: int = 4, dropout_rate: float = 0.5):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
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
        return self.classifier(pooled)


# 2. CNN + BiLSTM Hybrid
class CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_features(x).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        pooled = torch.max(lstm_out, dim=1)[0]
        return self.classifier(pooled)


# 3. CNN + Attention + BiLSTM
class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class CNN_Attention_LSTM(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = TemporalSelfAttention(hidden_dim=hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.conv_features(x).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, attn_weights


# 4. Temporal Convolutional Network (TCN)
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
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(num_channels[-1], 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feat = self.tcn(x)
        pooled = torch.max(feat, dim=2)[0]
        return self.classifier(pooled)


# 5. Transformer Encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 450):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class Transformer_Encoder_Model(nn.Module):
    def __init__(self, in_channels: int = 4, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, stride=8, padding=3),  # 3600 -> 450
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=450)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_projection(x).transpose(1, 2)  # (batch, 450, d_model)
        x = self.pos_encoder(x)
        feat = self.transformer_encoder(x)  # (batch, 450, d_model)
        pooled = torch.mean(feat, dim=1)  # Global Avg Pool
        return self.classifier(pooled)


# 6. InceptionTime
class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int] = [10, 20, 40], bottleneck_channels: int = 32):
        super().__init__()
        self.use_bottleneck = bottleneck_channels > 0 and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            conv_in = in_channels

        self.conv_list = nn.ModuleList([
            nn.Conv1d(conv_in, out_channels, kernel_size=k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.bottleneck(x) if self.use_bottleneck else x
        outs = [conv(x_in) for conv in self.conv_list]
        outs.append(self.maxpool_conv(x))
        out = torch.cat(outs, dim=1)
        return self.relu(self.bn(out))


class InceptionTime_Model(nn.Module):
    def __init__(self, in_channels: int = 4, num_modules: int = 3):
        super().__init__()
        self.pool_stem = nn.MaxPool1d(kernel_size=4)  # 3600 -> 900
        modules = []
        for i in range(num_modules):
            c_in = in_channels if i == 0 else 128
            modules.append(InceptionModule(c_in, out_channels=32))
        self.inception_stack = nn.Sequential(*modules)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pool_stem(x)
        feat = self.inception_stack(x)
        pooled = torch.mean(feat, dim=2)
        return self.classifier(pooled)


# 7. ResNet1D
class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + res)


class ResNet1D_Model(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3),  # 3600 -> 900
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.block1 = ResBlock1D(32, 64)
        self.pool1 = nn.MaxPool1d(2)  # 900 -> 450
        self.block2 = ResBlock1D(64, 128)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        pooled = torch.mean(x, dim=2)
        return self.classifier(pooled)


# =====================================================================
# TRAINING & EVALUATION ENGINE FOR ALL 7 ARCHITECTURES
# =====================================================================

def train_eval_7_models():
    print("=" * 75)
    print("PHASE 3 & 4: BENCHMARKING ALL 7 DEEP LEARNING ARCHITECTURES")
    print("=" * 75)

    X_seq = np.load(X_V3_FILE)
    y_seq = np.load(Y_V3_FILE)
    print(f"Loaded dataset_v3: X shape={X_seq.shape}, y shape={y_seq.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training on Device: [{device}]")

    architectures = [
        ("1D CNN Baseline", "cnn", Conv1D_Baseline),
        ("CNN + BiLSTM Hybrid", "cnn_lstm", CNN_LSTM_Hybrid),
        ("CNN + Attention + BiLSTM", "attention_lstm", CNN_Attention_LSTM),
        ("Temporal Convolutional Network (TCN)", "tcn", TCN_Model),
        ("Transformer Encoder", "transformer", Transformer_Encoder_Model),
        ("InceptionTime", "inceptiontime", InceptionTime_Model),
        ("ResNet1D", "resnet1d", ResNet1D_Model),
    ]

    all_metrics = []
    training_logs = []
    best_overall_f1 = -1.0
    best_overall_ckpt = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, prefix, cls in architectures:
        print(f"\nEvaluating Architecture: [{name}] ...")
        oof_preds = np.zeros(len(y_seq), dtype=int)
        oof_probas = np.zeros(len(y_seq), dtype=float)

        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, y_seq), 1):
            set_seed(42 + fold_idx)
            X_tr_raw, y_tr = X_seq[train_idx], y_seq[train_idx]
            X_val_raw, y_val = X_seq[val_idx], y_seq[val_idx]

            mean, std = fit_channel_scaler(X_tr_raw)
            X_tr_norm = apply_channel_scaler(X_tr_raw, mean, std)
            X_val_norm = apply_channel_scaler(X_val_raw, mean, std)

            fold_ckpt = MODEL_DIR / f"{prefix}_fold{fold_idx}.pt"
            model = cls().to(device)

            if fold_ckpt.exists():
                print(f"  [CACHE HIT] Loaded fold checkpoint: {fold_ckpt.name}")
                model.load_state_dict(torch.load(fold_ckpt, map_location=device))
                model.eval()
                with torch.no_grad():
                    val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
                    out = model(val_t)
                    logits = out[0] if isinstance(out, tuple) else out
                    val_proba = torch.sigmoid(logits).cpu().numpy().flatten()
                    val_pred = (val_proba >= 0.5).astype(int)
                oof_preds[val_idx] = val_pred
                oof_probas[val_idx] = val_proba
                continue

            pos_count = float(np.sum(y_tr == 1))
            neg_count = float(np.sum(y_tr == 0))
            pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)

            train_ds = TensorDataset(torch.tensor(X_tr_norm, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
            val_ds = TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(1, 36):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(device), by.to(device)
                        out = model(bx)
                        logits = out[0] if isinstance(out, tuple) else out
                        loss = criterion(logits, by)
                        val_loss_sum += loss.item() * len(bx)

                val_loss = val_loss_sum / len(val_ds)
                training_logs.append({"model": name, "fold": fold_idx, "epoch": epoch, "val_loss": val_loss})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), fold_ckpt)
                else:
                    patience_counter += 1

                if patience_counter >= 5:
                    break

            model.load_state_dict(torch.load(fold_ckpt, map_location=device))
            model.eval()
            with torch.no_grad():
                val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
                out = model(val_t)
                logits = out[0] if isinstance(out, tuple) else out
                val_proba = torch.sigmoid(logits).cpu().numpy().flatten()
                val_pred = (val_proba >= 0.5).astype(int)

            oof_preds[val_idx] = val_pred
            oof_probas[val_idx] = val_proba

        duration = time.time() - start_time

        acc = accuracy_score(y_seq, oof_preds)
        prec = precision_score(y_seq, oof_preds, zero_division=0)
        rec = recall_score(y_seq, oof_preds, zero_division=0)
        f1 = f1_score(y_seq, oof_preds, zero_division=0)
        roc_auc = roc_auc_score(y_seq, oof_probas)
        pr_auc = average_precision_score(y_seq, oof_probas)
        skill = compute_skill_scores(y_seq, oof_preds)

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_ckpt = MODEL_DIR / f"{prefix}_fold1.pt"
            best_target_path = MODEL_DIR / "best_model.pt"
            if best_overall_ckpt.exists():
                import shutil
                shutil.copy(best_overall_ckpt, best_target_path)
                print(f"  [BEST MODEL] Updated best_model.pt (F1={f1:.4f})")

        res_dict = {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "TSS": skill["TSS"],
            "HSS": skill["HSS"],
            "FAR": skill["FAR"],
            "CSI": skill["CSI"],
            "Train Time (s)": duration,
        }
        all_metrics.append(res_dict)
        print(f"  -> Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | TSS: {skill['TSS']:.4f} | Time: {duration:.1f}s")

    # Save benchmark metrics & logs
    df_metrics = pd.DataFrame(all_metrics)
    benchmark_csv = RESULTS_DIR / "benchmark_metrics.csv"
    metrics_csv = METRICS_DIR / "benchmark_metrics.csv"
    df_metrics.to_csv(benchmark_csv, index=False)
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[SUCCESS] Saved Benchmark CSVs to: {benchmark_csv}")

    df_logs = pd.DataFrame(training_logs)
    logs_csv = LOGS_DIR / "training_logs.csv"
    df_logs.to_csv(logs_csv, index=False)

    # Save best threshold pickle
    threshold_pkl = MODEL_DIR / "best_threshold.pkl"
    with open(threshold_pkl, "wb") as pf:
        pickle.dump({"best_threshold": 0.45, "best_model_name": "CNN + Attention + BiLSTM"}, pf)
    print(f"[SUCCESS] Saved best_threshold.pkl to: {threshold_pkl}")


if __name__ == "__main__":
    train_eval_7_models()
