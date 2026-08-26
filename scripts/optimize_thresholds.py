"""Task 2 & Task 3: Decision Threshold Optimization & Solar Flare Forecasting Metrics.

Sweeps thresholds from 0.05 to 0.95 in 0.05 increments.
Computes:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Solar Flare Forecasting Skill Scores:
  - True Skill Statistic (TSS = POD - POFD)
  - Heidke Skill Score (HSS)
  - Probability of Detection (POD = TP / (TP + FN))
  - False Alarm Ratio (FAR = FP / (TP + FP))
  - Critical Success Index (CSI = TP / (TP + FP + FN))

Generates:
- results/threshold_optimization_report.md
- results/solar_forecast_metrics_report.md
- results/threshold_metrics.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
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

OOF_FILE = RESULTS_DIR / "model_oof_predictions.npz"

THRESH_REPORT_MD = RESULTS_DIR / "threshold_optimization_report.md"
THRESH_ROOT_MD = PROJECT_ROOT / "threshold_optimization_report.md"

FORECAST_REPORT_MD = RESULTS_DIR / "solar_forecast_metrics_report.md"
FORECAST_ROOT_MD = PROJECT_ROOT / "solar_forecast_metrics_report.md"

THRESH_CSV = RESULTS_DIR / "threshold_metrics.csv"
THRESH_ROOT_CSV = PROJECT_ROOT / "threshold_metrics.csv"


def compute_metrics_at_threshold(y_true: np.ndarray, y_probas: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute all evaluation metrics and skill scores at a given classification threshold."""
    y_pred = (y_probas >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

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

    return {
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "TSS": tss,
        "HSS": hss,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def main():
    print("=" * 75)
    print("TASK 2 & 3: DECISION THRESHOLD OPTIMIZATION & FORECASTING SKILL SCORES")
    print("=" * 75)

    if not OOF_FILE.exists():
        print(f"[ERROR] OOF predictions file not found at {OOF_FILE}. Run train_architecture_comparison.py first!")
        return

    data = np.load(OOF_FILE)
    y_true = data["y_true"]
    y_probas = data["attn_probas"] if "attn_probas" in data else data["lstm_probas"]
    roc_auc = roc_auc_score(y_true, y_probas)

    thresholds = np.arange(0.05, 0.96, 0.05)
    records = []

    for t in thresholds:
        t_round = round(t, 2)
        m = compute_metrics_at_threshold(y_true, y_probas, t_round)
        records.append(m)

    df_thresh = pd.DataFrame(records)
    df_thresh["ROC-AUC"] = roc_auc

    # Find optimal thresholds
    opt_tss_row = df_thresh.loc[df_thresh["TSS"].idxmax()]
    opt_hss_row = df_thresh.loc[df_thresh["HSS"].idxmax()]
    opt_f1_row = df_thresh.loc[df_thresh["F1 Score"].idxmax()]

    # Default Operational Threshold: Maximize TSS (standard for space weather forecasting)
    opt_thresh = opt_tss_row["Threshold"]

    print(f"Optimal Threshold (Max TSS = {opt_tss_row['TSS']:.4f}): Threshold = {opt_thresh:.2f}")
    print(f"Optimal Threshold (Max HSS = {opt_hss_row['HSS']:.4f}): Threshold = {opt_hss_row['Threshold']:.2f}")
    print(f"Optimal Threshold (Max F1  = {opt_f1_row['F1 Score']:.4f}): Threshold = {opt_f1_row['Threshold']:.2f}")
    print("-" * 75)

    df_thresh.to_csv(THRESH_CSV, index=False)
    df_thresh.to_csv(THRESH_ROOT_CSV, index=False)
    print(f"[SUCCESS] Saved threshold metrics CSV to:\n  - {THRESH_CSV}\n  - {THRESH_ROOT_CSV}")

    # 1. threshold_optimization_report.md
    thresh_md = f"""# Decision Threshold Optimization Report for Solar Flare Forecasting

**Dataset**: Frozen HEL1OS Time-Series Tensor (`X_sequences_expanded.npy`)  
**Sample Count ($N$)**: `{len(y_true)}` sequences  
**Evaluated Threshold Range**: `0.05` to `0.95` (Step: `0.05`)  
**Optimal Operational Threshold**: **`{opt_thresh:.2f}`** (Maximizes True Skill Statistic TSS)  
**Report Location**: [threshold_optimization_report.md](file://{THRESH_REPORT_MD.resolve()})  

---

## 1. Threshold Optimization Metric Table

| Threshold | Accuracy | Precision | Recall (POD) | F1-Score | TSS | HSS | FAR | CSI | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for _, r in df_thresh.iterrows():
        is_opt = " **(Opt TSS)**" if r["Threshold"] == opt_thresh else ""
        thresh_md += f"| `{r['Threshold']:.2f}`{is_opt} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1 Score']:.4f} | **{r['TSS']:.4f}** | **{r['HSS']:.4f}** | {r['FAR']:.4f} | {r['CSI']:.4f} | {int(r['TP'])} | {int(r['FP'])} | {int(r['TN'])} | {int(r['FN'])} |\n"

    thresh_md += f"""
---

## 2. Optimal Operational Threshold Recommendations

1. **Space Weather Operational Standard (Max TSS)**:
   - **Recommended Threshold**: **`{opt_tss_row['Threshold']:.2f}`**
   - **TSS**: **`{opt_tss_row['TSS']:.4f}`** | **HSS**: **`{opt_tss_row['HSS']:.4f}`**
   - **POD (Recall)**: **`{opt_tss_row['POD']:.4f}`** | **FAR**: **`{opt_tss_row['FAR']:.4f}`**
2. **Balanced Detection (Max F1-Score)**:
   - **Recommended Threshold**: **`{opt_f1_row['Threshold']:.2f}`**
   - **F1-Score**: **`{opt_f1_row['F1 Score']:.4f}`** | **Precision**: **`{opt_f1_row['Precision']:.4f}`** | **Recall**: **`{opt_f1_row['Recall']:.4f}`**
"""

    THRESH_REPORT_MD.write_text(thresh_md, encoding="utf-8")
    THRESH_ROOT_MD.write_text(thresh_md, encoding="utf-8")
    print(f"[SUCCESS] Saved Threshold Optimization Report to: {THRESH_REPORT_MD}")

    # 2. solar_forecast_metrics_report.md
    forecast_md = f"""# Solar Flare Forecasting Skill Scores Report

**Report Location**: [solar_forecast_metrics_report.md](file://{FORECAST_REPORT_MD.resolve()})  
**Evaluation Standard**: Solar Physics & Space Weather Forecasting Skill Metrics  
**Optimal Decision Threshold**: `{opt_thresh:.2f}`  

---

## 1. Primary Operational Skill Scores (Threshold = {opt_thresh:.2f})

| Skill Metric Symbol | Metric Full Name | Metric Value | Benchmark Interpretation |
| :--- | :--- | :---: | :--- |
| **TSS** | **True Skill Statistic** | **`{opt_tss_row['TSS']:.4f}`** | Net operational skill over random chance ($[-1, +1]$ scale) |
| **HSS** | **Heidke Skill Score** | **`{opt_tss_row['HSS']:.4f}`** | Skill relative to random reference forecast ($[- \infty, +1]$ scale) |
| **POD** | **Probability of Detection (Sensitivity)** | **`{opt_tss_row['POD']:.4f}`** | Proportion of actual major flares correctly predicted |
| **FAR** | **False Alarm Ratio** | **`{opt_tss_row['FAR']:.4f}`** | Proportion of positive predictions that were false alarms |
| **CSI** | **Critical Success Index (Threat Score)** | **`{opt_tss_row['CSI']:.4f}`** | Ratio of TP to total positive forecast/event occurrences |

---

## 2. Mathematical Metric Formulations

$$ \\text{{TSS}} = \\text{{POD}} - \\text{{POFD}} = \\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FN}}}} - \\frac{{\\text{{FP}}}}{{\\text{{FP}} + \\text{{TN}}}} $$

$$ \\text{{HSS}} = \\frac{{2 (\\text{{TP}} \\cdot \\text{{TN}} - \\text{{FP}} \\cdot \\text{{FN}})}}{{(\\text{{TP}} + \\text{{FN}})(\\text{{FN}} + \\text{{TN}}) + (\\text{{TP}} + \\text{{FP}})(\\text{{FP}} + \\text{{TN}})}} $$

$$ \\text{{CSI}} = \\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FP}} + \\text{{FN}}}} $$
"""

    FORECAST_REPORT_MD.write_text(forecast_md, encoding="utf-8")
    FORECAST_ROOT_MD.write_text(forecast_md, encoding="utf-8")
    print(f"[SUCCESS] Saved Solar Forecast Metrics Report to: {FORECAST_REPORT_MD}")


if __name__ == "__main__":
    main()
