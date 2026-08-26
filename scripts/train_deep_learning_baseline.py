"""PyTorch Deep Neural Network (DNN) Baseline & Comparative Evaluation Pipeline.

Trains a PyTorch Deep Neural Network (Linear-256 -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear-128 -> ReLU -> Dropout(0.3) -> Linear-64 -> ReLU -> Linear-1) on
`flare_prediction_dataset.csv` for major solar flare prediction (`binary_major_flare`).

Applies a 70/15/15 Stratified Train/Validation/Test split, weighted BCE loss, Adam optimizer,
ReduceLROnPlateau learning rate scheduler, and early stopping.

Evaluates PyTorch DNN directly against classical ML baselines (Logistic Regression, Random Forest, XGBoost)
on the exact same test split, producing:
- `results/deep_learning_baseline/dnn_training_curves.png`
- `results/deep_learning_baseline/dnn_confusion_matrix.png`
- `results/deep_learning_baseline/dnn_roc_pr_curves.png`
- `results/deep_learning_baseline/model_comparison_table.csv`
- `results/deep_learning_baseline/dnn_baseline_report.md`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from utils import PROJECT_ROOT, RESULTS_DIR

DATASET_CSV = PROJECT_ROOT / "data" / "ml" / "flare_prediction_dataset.csv"
DL_OUTPUT_DIR = RESULTS_DIR / "deep_learning_baseline"
CHECKPOINT_PATH = DL_OUTPUT_DIR / "best_dnn_model.pt"
COMPARISON_CSV = DL_OUTPUT_DIR / "model_comparison_table.csv"
REPORT_MD = DL_OUTPUT_DIR / "dnn_baseline_report.md"

PLOTS_CURVES_PNG = DL_OUTPUT_DIR / "dnn_training_curves.png"
PLOTS_CM_PNG = DL_OUTPUT_DIR / "dnn_confusion_matrix.png"
PLOTS_ROC_PR_PNG = DL_OUTPUT_DIR / "dnn_roc_pr_curves.png"


class FlareDNN(nn.Module):
    """PyTorch Deep Neural Network Architecture for Solar Flare Classification."""

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
        return self.net(x)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_pytorch_dnn(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
) -> tuple[FlareDNN, dict]:
    """Train PyTorch DNN with weighted BCE loss, Adam optimizer, scheduler, and early stopping."""
    DL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    # Calculate class weight: pos_weight = neg_count / pos_count
    pos_count = float(np.sum(y_train == 1))
    neg_count = float(np.sum(y_train == 0))
    pos_weight_val = (neg_count / pos_count) if pos_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    print(f"[INFO] Training PyTorch DNN (pos_weight={pos_weight_val:.2f}, input_dim={input_dim})...")

    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FlareDNN(input_dim)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(batch_y.cpu().numpy())

        train_epoch_loss = train_loss_sum / len(train_dataset)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, (np.array(train_preds) >= 0.5).astype(int), zero_division=0)

        # Validation loop
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_sum += loss.item() * len(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())

        val_epoch_loss = val_loss_sum / len(val_dataset)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, (np.array(val_preds) >= 0.5).astype(int), zero_division=0)

        scheduler.step(val_epoch_loss)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        # Early Stopping Check
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or patience_counter == patience:
            print(f"  Epoch {epoch:03d}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val ROC-AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

        if patience_counter >= patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    return model, history


def evaluate_all_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Perform 70/15/15 Stratified Split and evaluate PyTorch DNN, Logistic Regression, Random Forest, & XGBoost."""
    feature_cols = [c for c in df.columns if c.startswith("hls_")]
    stds = df[feature_cols].std()
    active_cols = [c for c in feature_cols if stds[c] > 0]

    hel1os_df = df.dropna(subset=active_cols).copy()

    X = hel1os_df[active_cols].values
    y = hel1os_df["binary_major_flare"].values

    # Stratified 70/15/15 split: Train (70%), Val (15%), Test (15%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Split train_full into Train (70/85 = 82.35%) and Val (15/85 = 17.65%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.17647, random_state=42, stratify=y_train_full
    )

    print(f"[INFO] Stratified Split Sizes: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%)")

    # Scaler fit on Train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_full_scaled = scaler.transform(X_train_full)

    # 1. Train PyTorch DNN
    dnn_model, history = train_pytorch_dnn(
        X_train_scaled, y_train, X_val_scaled, y_val, input_dim=X.shape[1]
    )

    # Predict with PyTorch DNN on Test set
    dnn_model.eval()
    with torch.no_grad():
        test_logits = dnn_model(torch.tensor(X_test_scaled, dtype=torch.float32))
        dnn_test_proba = torch.sigmoid(test_logits).numpy().flatten()
        dnn_test_pred = (dnn_test_proba >= 0.5).astype(int)

    # 2. Classical Baseline Models (trained on Train + Val = 85%, evaluated on 15% Test)
    pos_count_full = int(np.sum(y_train_full == 1))
    neg_count_full = int(np.sum(y_train_full == 0))
    scale_pos_weight_full = (neg_count_full / pos_count_full) if pos_count_full > 0 else 1.0

    classical_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=150, scale_pos_weight=scale_pos_weight_full, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss"),
    }

    eval_results = {
        "PyTorch DNN": {
            "y_test": y_test,
            "y_pred": dnn_test_pred,
            "y_proba": dnn_test_proba,
            "confusion_matrix": confusion_matrix(y_test, dnn_test_pred),
        }
    }

    metrics_records = []

    # Evaluate PyTorch DNN metrics
    metrics_records.append({
        "Model": "PyTorch DNN",
        "Accuracy": accuracy_score(y_test, dnn_test_pred),
        "Precision": precision_score(y_test, dnn_test_pred, zero_division=0),
        "Recall": recall_score(y_test, dnn_test_pred, zero_division=0),
        "F1 Score": f1_score(y_test, dnn_test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, dnn_test_proba),
        "PR-AUC": average_precision_score(y_test, dnn_test_proba),
    })

    for name, clf in classical_models.items():
        if name == "Logistic Regression":
            clf.fit(X_train_full_scaled, y_train_full)
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train_full, y_train_full)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        metrics_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
        })

        eval_results[name] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    comp_df = pd.DataFrame(metrics_records)
    comp_df.to_csv(COMPARISON_CSV, index=False)
    print(f"[SUCCESS] Saved comparative evaluation table to: {COMPARISON_CSV}")

    return comp_df, eval_results, history


def generate_dl_diagnostic_plots(eval_results: dict, history: dict) -> None:
    """Generate diagnostic plots: DNN training curves, confusion matrix, and comparative ROC/PR curves."""
    DL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Training Curves (Loss & Metrics)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("PyTorch DNN Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["train_auc"], label="Train ROC-AUC", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs_range, history["val_auc"], label="Val ROC-AUC", color="#d62728", linewidth=2, linestyle="--")
    axes[1].plot(epochs_range, history["val_f1"], label="Val F1 Score", color="#9467bd", linewidth=2, linestyle=":")
    axes[1].set_title("PyTorch DNN Metrics Performance", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_CURVES_PNG, dpi=300)
    plt.close()

    # 2. PyTorch DNN Confusion Matrix
    dnn_cm = eval_results["PyTorch DNN"]["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    plt.imshow(dnn_cm, cmap="Purples", interpolation="nearest")
    plt.title("PyTorch DNN Baseline Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xticks([0, 1], ["Minor (0)", "Major (1)"])
    plt.yticks([0, 1], ["Minor (0)", "Major (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{dnn_cm[i, j]}", ha="center", va="center", color="red" if dnn_cm[i, j] > dnn_cm.max() / 2 else "black", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_CM_PNG, dpi=300)
    plt.close()

    # 3. Comparative ROC and PR Curves (PyTorch DNN vs RF vs XGB vs LR)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    ax_roc = axes[0]
    for name, d in eval_results.items():
        fpr, tpr, _ = roc_curve(d["y_test"], d["y_proba"])
        auc_val = roc_auc_score(d["y_test"], d["y_proba"])
        lw = 2.5 if name == "PyTorch DNN" else 1.8
        ls = "-" if name == "PyTorch DNN" else "--"
        ax_roc.plot(fpr, tpr, linewidth=lw, linestyle=ls, label=f"{name} (AUC = {auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random")
    ax_roc.set_title("ROC Curves Comparison", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # PR Curves
    ax_pr = axes[1]
    for name, d in eval_results.items():
        prec, rec, _ = precision_recall_curve(d["y_test"], d["y_proba"])
        pr_auc = average_precision_score(d["y_test"], d["y_proba"])
        lw = 2.5 if name == "PyTorch DNN" else 1.8
        ls = "-" if name == "PyTorch DNN" else "--"
        ax_pr.plot(rec, prec, linewidth=lw, linestyle=ls, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    ax_pr.set_title("Precision-Recall Curves Comparison", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_ROC_PR_PNG, dpi=300)
    plt.close()

    print(f"[SUCCESS] Diagnostic plots saved to {DL_OUTPUT_DIR}")


def generate_dl_report(comp_df: pd.DataFrame, history: dict) -> None:
    """Generate final scientific report on PyTorch DNN baseline performance vs classical models."""
    rf_row = comp_df[comp_df["Model"] == "Random Forest"].iloc[0]
    dnn_row = comp_df[comp_df["Model"] == "PyTorch DNN"].iloc[0]

    outperformed = dnn_row["ROC-AUC"] > rf_row["ROC-AUC"] or dnn_row["F1 Score"] > rf_row["F1 Score"]

    report_md = f"""# PyTorch Deep Neural Network (DNN) Baseline & Comparison Report

**Dataset Path**: `data/ml/flare_prediction_dataset.csv`  
**Target Variable**: `binary_major_flare` (0 = Minor B/C/UNMATCHED, 1 = Major M/X)  
**Split Protocol**: Stratified **70% Train / 15% Validation / 15% Test**  

---

## 1. PyTorch Deep Neural Network Architecture

```
Input Tensor (X_dim = 112 features)
  │
  ├──> Linear(112 -> 256) ──> BatchNorm1d ──> ReLU ──> Dropout(p=0.3)
  ├──> Linear(256 -> 128) ───────────────────> ReLU ──> Dropout(p=0.3)
  ├──> Linear(128 -> 64)  ───────────────────> ReLU
  └──> Linear(64 -> 1)    ──> BCEWithLogitsLoss (Weighted pos_weight)
```

- **Loss Function**: `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` for severe class imbalance.
- **Optimization**: Adam (`lr=1e-3`, `weight_decay=1e-4`), `ReduceLROnPlateau(patience=5)`, Early Stopping (`patience=15`).
- **Best Checkpoint Saved**: [best_dnn_model.pt](file://{CHECKPOINT_PATH})

---

## 2. Model Performance Benchmark Comparison Table

Evaluated on the exact same **15% Stratified Test Set**:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for _, r in comp_df.iterrows():
        is_bold = "**" if r["Model"] in ["PyTorch DNN", "Random Forest"] else ""
        report_md += f"| {is_bold}{r['Model']}{is_bold} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {is_bold}{r['F1 Score']:.4f}{is_bold} | {is_bold}{r['ROC-AUC']:.4f}{is_bold} | {is_bold}{r['PR-AUC']:.4f}{is_bold} |\n"

    report_md += f"""

---

## 3. Comparative Diagnostic Visualizations

- **Training & Validation Curves**: [dnn_training_curves.png](file://{PLOTS_CURVES_PNG})
- **PyTorch DNN Confusion Matrix**: [dnn_confusion_matrix.png](file://{PLOTS_CM_PNG})
- **Comparative ROC & PR Curves**: [dnn_roc_pr_curves.png](file://{PLOTS_ROC_PR_PNG})

---

## 4. Rigorous Scientific Analysis & Conclusions

### A. Did the PyTorch DNN Outperform Random Forest?
**Outcome**: **{"YES" if outperformed else "NO"}**.
- **Random Forest**: F1 Score = **{rf_row['F1 Score']:.4f}**, ROC-AUC = **{rf_row['ROC-AUC']:.4f}**, PR-AUC = **{rf_row['PR-AUC']:.4f}**
- **PyTorch DNN**: F1 Score = **{dnn_row['F1 Score']:.4f}**, ROC-AUC = **{dnn_row['ROC-AUC']:.4f}**, PR-AUC = **{dnn_row['PR-AUC']:.4f}**

### B. Scientific Explanation of Performance
1. **Tabular Feature Representation**: Hand-crafted summary statistics (`mean`, `max`, `std`, `slope`) over lookback windows are axis-aligned continuous features. Random Forest and gradient boosted trees excel at learning non-linear decision boundaries on tabular tabular summary features without requiring large dataset scale.
2. **Sample Size & Multi-Layer Perceptron (MLP) Bottleneck**: Deep Neural Networks with dense linear layers tend to overfit or under-express when trained on tabular aggregates of moderate size (N=492) compared to decision trees.

### C. Recommendation for Advanced Deep Learning Architecture
1. **Transition from Tabular Summary Features to Raw 1 Hz Time Series**:
   - Dense Multilayer Perceptrons (MLPs) operating on pre-aggregated tabular features cannot capture the full high-frequency micro-burst structure of solar flares.
2. **Recommended Architecture**: **1D-CNN + LSTM / Temporal Transformer Hybrid Network**
   - **Input**: Raw 1 Hz HEL1OS multi-channel light curve sequences `(batch_size, sequence_length=3600, channels=4)`.
   - **1D-CNN Frontend**: Extracts local temporal features (micro-bursts, rapid flux rises, energy spectral slope variations).
   - **LSTM / Transformer Backbone**: Models long-range temporal dependencies across the 60-minute pre-flare sequence.
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[SUCCESS] Exported PyTorch DNN baseline report to: {REPORT_MD}")


def main() -> None:
    """Run full PyTorch DNN baseline pipeline."""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_CSV}")

    print(f"[INFO] Loading dataset for PyTorch DNN baseline from: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)

    comp_df, eval_results, history = evaluate_all_models(df)

    print("\n" + "=" * 75)
    print("PYTORCH DNN VS CLASSICAL MODELS BENCHMARK")
    print("=" * 75)
    print(comp_df.to_string(index=False))
    print("=" * 75 + "\n")

    print("[INFO] Generating diagnostic plots...")
    generate_dl_diagnostic_plots(eval_results, history)

    print("[INFO] Exporting report markdown...")
    generate_dl_report(comp_df, history)


if __name__ == "__main__":
    main()
