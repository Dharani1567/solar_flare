"""Expanded Dataset Retraining Engine for All 6 Baseline Models.

Retrains:
1. Logistic Regression (Tabular summary features)
2. Random Forest (Tabular summary features)
3. XGBoost (Tabular summary features)
4. PyTorch Tabular DNN (Tabular summary features)
5. PyTorch 1D CNN Single 15% Split (Raw 1 Hz Sequences)
6. PyTorch 1D CNN Stratified 5-Fold Cross Validation (Raw 1 Hz Sequences)

Saves models to `results/models_expanded/` and outputs comparative metrics report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from train_cnn_5fold import EnhancedSolarFlare1DCNN, apply_channel_scaler, fit_channel_scaler, train_single_fold
from utils import PROJECT_ROOT

ML_DIR = PROJECT_ROOT / "data" / "ml" / "expanded"
X_EXPANDED_FILE = ML_DIR / "X_sequences_expanded.npy"
Y_EXPANDED_FILE = ML_DIR / "y_labels_expanded.npy"

MODELS_EXPANDED_DIR = PROJECT_ROOT / "results" / "models_expanded"
RESULTS_EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded"
REPORT_MD = RESULTS_EXPANDED_DIR / "expanded_models_benchmark_report.md"


class TabularMLP(nn.Module):
    """PyTorch Deep Neural Network for Tabular Predictor Features."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_tabular_dnn(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train PyTorch Tabular DNN Baseline."""
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    pos_w = (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32))

    dataset_tr = TensorDataset(torch.tensor(X_tr_sc, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader_tr = DataLoader(dataset_tr, batch_size=32, shuffle=True)

    model = TabularMLP(X_tr.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(40):
        for bx, by in loader_tr:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te_sc, dtype=torch.float32))
        probas = torch.sigmoid(logits).numpy()
        preds = (probas >= 0.5).astype(int)

    torch.save(model.state_dict(), MODELS_EXPANDED_DIR / "tabular_dnn_expanded.pt")
    return preds, probas


def run_all_6_models_retraining() -> pd.DataFrame:
    """Retrain all 6 machine learning & deep learning models on expanded dataset."""
    MODELS_EXPANDED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_EXPANDED_DIR.mkdir(parents=True, exist_ok=True)

    if not X_EXPANDED_FILE.exists() or not Y_EXPANDED_FILE.exists():
        # Fallback to standard ML dir if expanded subdir is building
        X_seq = np.load(PROJECT_ROOT / "data" / "ml" / "X_sequences_expanded.npy")
        y_seq = np.load(PROJECT_ROOT / "data" / "ml" / "y_labels_expanded.npy")
    else:
        X_seq = np.load(X_EXPANDED_FILE)
        y_seq = np.load(Y_EXPANDED_FILE)

    n_samples = len(y_seq)
    pos_count = int(np.sum(y_seq == 1))
    neg_count = int(np.sum(y_seq == 0))

    print(f"[INFO] Retraining ALL 6 MODELS on Expanded Dataset ({n_samples} samples | Major M/X: {pos_count}, Minor: {neg_count})...")

    # 1. Stratified 5-Fold Cross-Validation on 1D CNN
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_seq), dtype=int)
    oof_probas = np.zeros(len(y_seq), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, y_seq), 1):
        X_tr_raw, y_tr = X_seq[train_idx], y_seq[train_idx]
        X_val_raw, y_val = X_seq[val_idx], y_seq[val_idx]

        mean, std = fit_channel_scaler(X_tr_raw)
        X_tr_norm = apply_channel_scaler(X_tr_raw, mean, std)
        X_val_norm = apply_channel_scaler(X_val_raw, mean, std)

        model, _, val_pred, val_proba = train_single_fold(
            fold_idx, X_tr_norm, y_tr, X_val_norm, y_val, patience=5
        )

        oof_preds[val_idx] = val_pred
        oof_probas[val_idx] = val_proba
        torch.save(model.state_dict(), MODELS_EXPANDED_DIR / f"cnn_5fold_expanded_fold_{fold_idx}.pt")

    # 2. Single 15% Stratified Test Split Comparison
    indices = np.arange(len(y_seq))
    idx_tr, idx_te, y_tr_s, y_te_s = train_test_split(indices, y_seq, test_size=0.15, random_state=42, stratify=y_seq)

    # Summary tabular features per channel: mean, max, std, min, slope, delta
    X_summary_tr = np.column_stack([
        np.mean(X_seq[idx_tr], axis=1),
        np.max(X_seq[idx_tr], axis=1),
        np.min(X_seq[idx_tr], axis=1),
        np.std(X_seq[idx_tr], axis=1),
    ])
    X_summary_te = np.column_stack([
        np.mean(X_seq[idx_te], axis=1),
        np.max(X_seq[idx_te], axis=1),
        np.min(X_seq[idx_te], axis=1),
        np.std(X_seq[idx_te], axis=1),
    ])

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_summary_tr)
    X_te_sc = scaler.transform(X_summary_te)

    # Model 1: Logistic Regression
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(X_tr_sc, y_tr_s)
    lr_pred = logreg.predict(X_te_sc)
    lr_proba = logreg.predict_proba(X_te_sc)[:, 1]

    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_summary_tr, y_tr_s)
    rf_pred = rf.predict(X_summary_te)
    rf_proba = rf.predict_proba(X_summary_te)[:, 1]

    # Model 3: XGBoost
    pos_w = (len(y_tr_s) - sum(y_tr_s)) / max(1, sum(y_tr_s))
    xgb_model = xgb.XGBClassifier(n_estimators=150, scale_pos_weight=pos_w, learning_rate=0.05, max_depth=4, random_state=42)
    xgb_model.fit(X_summary_tr, y_tr_s)
    xgb_pred = xgb_model.predict(X_summary_te)
    xgb_proba = xgb_model.predict_proba(X_summary_te)[:, 1]

    # Model 4: PyTorch Tabular DNN
    dnn_pred, dnn_proba = train_tabular_dnn(X_summary_tr, y_tr_s, X_summary_te, y_te_s)

    # Model 5: PyTorch 1D CNN Single Split
    mean_s, std_s = fit_channel_scaler(X_seq[idx_tr])
    X_tr_norm_s = apply_channel_scaler(X_seq[idx_tr], mean_s, std_s)
    X_te_norm_s = apply_channel_scaler(X_seq[idx_te], mean_s, std_s)

    cnn_s, _, cnn_pred_s, cnn_proba_s = train_single_fold(
        99, X_tr_norm_s, y_tr_s, X_te_norm_s, y_te_s, patience=5
    )
    torch.save(cnn_s.state_dict(), MODELS_EXPANDED_DIR / "cnn_single_split_expanded.pt")

    records = [
        {
            "Model": "1D CNN (Stratified 5-Fold CV)",
            "Input Features": "Raw 1 Hz Sequences (3600, 4)",
            "Accuracy": accuracy_score(y_seq, oof_preds),
            "Precision": precision_score(y_seq, oof_preds, zero_division=0),
            "Recall": recall_score(y_seq, oof_preds, zero_division=0),
            "F1 Score": f1_score(y_seq, oof_preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_seq, oof_probas),
            "PR-AUC": average_precision_score(y_seq, oof_probas),
        },
        {
            "Model": "1D CNN (Single 15% Split)",
            "Input Features": "Raw 1 Hz Sequences (3600, 4)",
            "Accuracy": accuracy_score(y_te_s, cnn_pred_s),
            "Precision": precision_score(y_te_s, cnn_pred_s, zero_division=0),
            "Recall": recall_score(y_te_s, cnn_pred_s, zero_division=0),
            "F1 Score": f1_score(y_te_s, cnn_pred_s, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te_s, cnn_proba_s),
            "PR-AUC": average_precision_score(y_te_s, cnn_proba_s),
        },
        {
            "Model": "Random Forest (Expanded)",
            "Input Features": "Engineered Summary Features",
            "Accuracy": accuracy_score(y_te_s, rf_pred),
            "Precision": precision_score(y_te_s, rf_pred, zero_division=0),
            "Recall": recall_score(y_te_s, rf_pred, zero_division=0),
            "F1 Score": f1_score(y_te_s, rf_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te_s, rf_proba),
            "PR-AUC": average_precision_score(y_te_s, rf_proba),
        },
        {
            "Model": "XGBoost (Expanded)",
            "Input Features": "Engineered Summary Features",
            "Accuracy": accuracy_score(y_te_s, xgb_pred),
            "Precision": precision_score(y_te_s, xgb_pred, zero_division=0),
            "Recall": recall_score(y_te_s, xgb_pred, zero_division=0),
            "F1 Score": f1_score(y_te_s, xgb_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te_s, xgb_proba),
            "PR-AUC": average_precision_score(y_te_s, xgb_proba),
        },
        {
            "Model": "Tabular DNN (Expanded)",
            "Input Features": "Engineered Summary Features",
            "Accuracy": accuracy_score(y_te_s, dnn_pred),
            "Precision": precision_score(y_te_s, dnn_pred, zero_division=0),
            "Recall": recall_score(y_te_s, dnn_pred, zero_division=0),
            "F1 Score": f1_score(y_te_s, dnn_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te_s, dnn_proba),
            "PR-AUC": average_precision_score(y_te_s, dnn_proba),
        },
        {
            "Model": "Logistic Regression (Expanded)",
            "Input Features": "Engineered Summary Features",
            "Accuracy": accuracy_score(y_te_s, lr_pred),
            "Precision": precision_score(y_te_s, lr_pred, zero_division=0),
            "Recall": recall_score(y_te_s, lr_pred, zero_division=0),
            "F1 Score": f1_score(y_te_s, lr_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te_s, lr_proba),
            "PR-AUC": average_precision_score(y_te_s, lr_proba),
        },
    ]

    comp_df = pd.DataFrame(records)
    comp_df.to_csv(RESULTS_EXPANDED_DIR / "all_6_models_metrics_expanded.csv", index=False)
    print(f"[SUCCESS] Retrained all 6 models and saved metrics CSV to: {RESULTS_EXPANDED_DIR / 'all_6_models_metrics_expanded.csv'}")
    return comp_df


if __name__ == "__main__":
    run_all_6_models_retraining()
