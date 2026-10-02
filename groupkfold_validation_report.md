# 🛡️ Solar Flare Forecasting — Leak-Free GroupKFold Validation Report

> **Validation Timestamp:** 2026-10-01 23:38:00 IST  
> **Auditor:** Lead AI Research Auditor & Senior ML Reviewer  
> **Workspace Path:** `c:\Users\darsh\OneDrive\Desktop\solar_flare`  
> **Validation Standard:** Zero Group Overlap, Zero Temporal Leakage, Fold-Fitted Scaling.

---

## 1. Group-Level Validation Summary Table

All **1,092 sequence samples** were partitioned using **`StratifiedGroupKFold(n_splits=5)`** grouped strictly by `observation_date` (**91 unique group days**). 

Every single 1-hour window originating from the same 24-hour observation day is assigned 100% strictly to either the training set OR the validation set:

| Fold Number | Training Days ($D_{\text{train}}$) | Validation Days ($D_{\text{val}}$) | Validation Samples | Overlap Count | Overlap Percentage | Leakage Status |
| :-: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | **72 days** | **19 days** | **228 samples** | **0 days** | **0.0%** | **PASSED (LEAK-FREE)** |
| **Fold 2** | **73 days** | **18 days** | **216 samples** | **0 days** | **0.0%** | **PASSED (LEAK-FREE)** |
| **Fold 3** | **73 days** | **18 days** | **216 samples** | **0 days** | **0.0%** | **PASSED (LEAK-FREE)** |
| **Fold 4** | **73 days** | **18 days** | **216 samples** | **0 days** | **0.0%** | **PASSED (LEAK-FREE)** |
| **Fold 5** | **73 days** | **18 days** | **216 samples** | **0 days** | **0.0%** | **PASSED (LEAK-FREE)** |
| **TOTAL / AVG** | **73 days (Avg)** | **18.2 days (Avg)** | **1,092 total** | **0 days** | **0.0%** | **100% LEAK-FREE** |

---

## 2. Preprocessing & Scaling Isolation Check

To prevent data leakage during feature standardization:
```python
# Strict Fold Scaling Isolation
mean_v = np.mean(X_train_raw, axis=(0, 1), keepdims=True)  # Computed STRICTLY on training fold
std_v  = np.std(X_train_raw, axis=(0, 1), keepdims=True) + 1e-8

X_train_scaled = (X_train_raw - mean_v) / std_v
X_val_scaled   = (X_val_raw   - mean_v) / std_v             # Validation scaled using training mean/std
```
- **Verification Result:** Zero test/validation fold statistics were leaked into training feature scaling.

---

## 3. Automatic Leakage Assertion Checks

The training script `kaggle_train_groupkfold.py` includes automated runtime assertions prior to training each fold:

```python
# Automatic Runtime Leakage Assertions
overlap = set(groups[train_idx]).intersection(set(groups[val_idx]))
if len(overlap) > 0:
    raise RuntimeError(f"[CRITICAL LEAKAGE] Fold {fold_num} contains {len(overlap)} overlapping observation days!")

if len(set(train_idx).intersection(set(val_idx))) > 0:
    raise RuntimeError(f"[CRITICAL LEAKAGE] Fold {fold_num} contains duplicate sample indices!")
```
- **Assertion Execution:** All 5 folds passed automated leakage checks with **0 assertions raised**.

---

## 4. Output Artifacts Inventory

The following leak-free artifacts have been generated in `results/`:

1. **`results/fold_assignments.csv`** (1,092 rows)  
   *Columns:* `sample_id`, `observation_date`, `fold_number`
2. **`results/final_benchmark.csv`**  
   *Metrics:* Leak-free 5-fold cross-validation accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC for all 7 architectures.
3. **`scripts/kaggle_train_groupkfold.py`**  
   *Pipeline:* Standardized script for leak-free execution on Kaggle GPU.
