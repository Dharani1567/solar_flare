# Comprehensive Confusion Matrix & Detection Metrics Report

**Dataset**: Frozen HEL1OS Time-Series Tensor (`X_sequences_expanded.npy`)  
**Sample Count ($N$)**: `732` sequences  
**Report Location**: [confusion_matrix_report.md](file:///home/dharani/Desktop/solar_flare/results/confusion_matrix_report.md)  

---

## 1. Aggregated Out-Of-Fold Confusion Matrix

| | Predicted Minor Flare ($y=0$) | Predicted Major Flare ($y=1$) | Total Actual Events | Recall / Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Minor Flare ($y=0$)** | **557** (True Negatives) | **61** (False Positives) | 618 | **0.9013** (Specificity) |
| **Actual Major Flare ($y=1$)** | **41** (False Negatives) | **73** (True Positives) | 114 | **0.6404** (Sensitivity / POD) |
| **Total Predicted** | 598 | 134 | 732 | **0.8607** (Accuracy) |

---

## 2. Key Metrics Summary

- **Accuracy**: `0.8607`
- **Precision**: `0.5448`
- **Recall (Probability of Detection - POD)**: `0.6404`
- **F1-Score**: `0.5887`
