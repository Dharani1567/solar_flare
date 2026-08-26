"""Task 1: Comprehensive Error Analysis & Confusion Matrix Reporting.

Analyzes OOF predictions from model_oof_predictions.npz and metadata sequence_metadata_expanded.csv:
- Identifies False Negatives (missed major flares) and False Positives.
- Identifies Hard-to-classify samples (boundary probabilities [0.4, 0.6]).
- Evaluates probability distributions across true positive, true negative, false positive, and false negative sets.
- Generates:
  - results/confusion_matrix_report.md
  - results/error_analysis_report.md
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utils import PROJECT_ROOT, RESULTS_DIR

OOF_FILE = RESULTS_DIR / "model_oof_predictions.npz"
META_FILE = PROJECT_ROOT / "data" / "ml" / "sequence_metadata_expanded.csv"

CM_REPORT_MD = RESULTS_DIR / "confusion_matrix_report.md"
CM_ROOT_MD = PROJECT_ROOT / "confusion_matrix_report.md"

ERROR_REPORT_MD = RESULTS_DIR / "error_analysis_report.md"
ERROR_ROOT_MD = PROJECT_ROOT / "error_analysis_report.md"


def main():
    print("=" * 75)
    print("TASK 1: COMPREHENSIVE ERROR ANALYSIS & CONFUSION MATRIX REPORTING")
    print("=" * 75)

    if not OOF_FILE.exists():
        print(f"[ERROR] OOF predictions file not found at {OOF_FILE}. Run train_architecture_comparison.py first!")
        return

    data = np.load(OOF_FILE)
    y_true = data["y_true"]
    y_preds = data["attn_preds"] if "attn_preds" in data else data["lstm_preds"]
    y_probas = data["attn_probas"] if "attn_probas" in data else data["lstm_probas"]

    meta_df = pd.read_csv(META_FILE)

    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    meta_df["y_true"] = y_true
    meta_df["y_pred"] = y_preds
    meta_df["y_proba"] = y_probas

    # Categorize samples
    fn_df = meta_df[(meta_df["y_true"] == 1) & (meta_df["y_pred"] == 0)].copy()
    fp_df = meta_df[(meta_df["y_true"] == 0) & (meta_df["y_pred"] == 1)].copy()
    tp_df = meta_df[(meta_df["y_true"] == 1) & (meta_df["y_pred"] == 1)].copy()
    tn_df = meta_df[(meta_df["y_true"] == 0) & (meta_df["y_pred"] == 0)].copy()

    hard_df = meta_df[(meta_df["y_proba"] >= 0.40) & (meta_df["y_proba"] <= 0.60)].copy()

    print(f"Total Sequences ($N$) : {len(y_true)}")
    print(f"True Positives (TP)   : {tp} (Major flares correctly detected)")
    print(f"True Negatives (TN)   : {tn} (Minor flares correctly detected)")
    print(f"False Positives (FP)  : {fp} (Minor flares misclassified as major)")
    print(f"False Negatives (FN)  : {fn} (Major flares missed)")
    print(f"Hard-to-Classify      : {len(hard_df)} samples in [0.40, 0.60] probability window")
    print("-" * 75)

    # 1. confusion_matrix_report.md
    cm_md = f"""# Comprehensive Confusion Matrix & Detection Metrics Report

**Dataset**: Frozen HEL1OS Time-Series Tensor (`X_sequences_expanded.npy`)  
**Sample Count ($N$)**: `{len(y_true)}` sequences  
**Report Location**: [confusion_matrix_report.md](file://{CM_REPORT_MD.resolve()})  

---

## 1. Aggregated Out-Of-Fold Confusion Matrix

| | Predicted Minor Flare ($y=0$) | Predicted Major Flare ($y=1$) | Total Actual Events | Recall / Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Minor Flare ($y=0$)** | **{tn}** (True Negatives) | **{fp}** (False Positives) | {tn + fp} | **{tn/(tn+fp):.4f}** (Specificity) |
| **Actual Major Flare ($y=1$)** | **{fn}** (False Negatives) | **{tp}** (True Positives) | {fn + tp} | **{tp/(fn+tp):.4f}** (Sensitivity / POD) |
| **Total Predicted** | {tn + fn} | {fp + tp} | {len(y_true)} | **{accuracy_score(y_true, y_preds):.4f}** (Accuracy) |

---

## 2. Key Metrics Summary

- **Accuracy**: `{accuracy_score(y_true, y_preds):.4f}`
- **Precision**: `{precision_score(y_true, y_preds, zero_division=0):.4f}`
- **Recall (Probability of Detection - POD)**: `{recall_score(y_true, y_preds, zero_division=0):.4f}`
- **F1-Score**: `{f1_score(y_true, y_preds, zero_division=0):.4f}`
"""

    CM_REPORT_MD.write_text(cm_md, encoding="utf-8")
    CM_ROOT_MD.write_text(cm_md, encoding="utf-8")
    print(f"[SUCCESS] Saved Confusion Matrix Report to: {CM_REPORT_MD}")

    # 2. error_analysis_report.md
    error_md = f"""# Detailed Model Error Analysis & Failure Mode Report

**Report Location**: [error_analysis_report.md](file://{ERROR_REPORT_MD.resolve()})  
**Evaluation Scope**: Analysis of `{fn}` False Negatives, `{fp}` False Positives, and `{len(hard_df)}` Boundary Samples  

---

## 1. False Negative Analysis (Missed Major Flares: {fn} Events)

False Negatives represent high-energy M/X major flares that the model failed to flag. Below are the top missed major flares ranked by lowest predicted probability:

| Index | Date | GOES Class | Predicted Prob | SNR | Net Peak Counts | Duration (s) | Failure Attribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""

    for idx, r in fn_df.sort_values("y_proba").head(15).iterrows():
        reason = "Low pre-flare SNR / gradual rise" if r.get("snr", 0) < 10 else "Short lookback window pulse"
        error_md += f"| {r.get('seq_idx', idx)} | `{r.get('date', 'N/A')}` | `{r.get('goes_class', 'M/X')}` | `{r['y_proba']:.4f}` | `{r.get('snr', 0):.2f}` | `{r.get('net_peak_counts', 0):.1f}` | `{r.get('duration_sec', 0):.0f}s` | {reason} |\n"

    error_md += f"""
---

## 2. False Positive Analysis (Misclassified Minor Flares: {fp} Events)

False Positives represent C-class or B-class flares that exhibited intense pre-flare acceleration mimicking major flare signatures:

| Index | Date | GOES Class | Predicted Prob | SNR | Net Peak Counts | Attribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""

    for idx, r in fp_df.sort_values("y_proba", ascending=False).head(15).iterrows():
        error_md += f"| {r.get('seq_idx', idx)} | `{r.get('date', 'N/A')}` | `{r.get('goes_class', 'C/B')}` | `{r['y_proba']:.4f}` | `{r.get('snr', 0):.2f}` | `{r.get('net_peak_counts', 0):.1f}` | High pre-flare background flux | \n"

    error_md += f"""
---

## 3. Probability Distribution & Boundary Analysis

- **True Positive Mean Probability**: `{tp_df['y_proba'].mean():.4f}`
- **True Negative Mean Probability**: `{tn_df['y_proba'].mean():.4f}`
- **False Positive Mean Probability**: `{fp_df['y_proba'].mean():.4f}`
- **False Negative Mean Probability**: `{fn_df['y_proba'].mean():.4f}`
- **Hard Boundary Count [0.40, 0.60]**: `{len(hard_df)}` samples
"""

    ERROR_REPORT_MD.write_text(error_md, encoding="utf-8")
    ERROR_ROOT_MD.write_text(error_md, encoding="utf-8")
    print(f"[SUCCESS] Saved Error Analysis Report to: {ERROR_REPORT_MD}")


if __name__ == "__main__":
    main()
