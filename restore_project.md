# 🔄 Solar Flare Forecasting — Project Asset Restoration Guide

> **Document Version:** 1.0  
> **Purpose:** Step-by-step instructions to 100% regenerate any deleted asset, model checkpoint, or dataset archive removed during storage cleanup.

---

## 1. Quick Restoration Command Summary

| Deleted Asset | Storage Saved | Regeneration Command / Method | Execution Time |
| :--- | :---: | :--- | :---: |
| `data/ml/X_sequences_expanded.*` | **80.54 MB** | `python scripts/expand_dataset.py` (or duplicate copy of `v3`) | < 2 seconds |
| `solar_flare_dataset_v3.zip` | **34.71 MB** | `python scripts/prepare_kaggle_dataset.py` | ~ 3 seconds |
| `pradan1.issdc.gov.in/.../*.part` | **32.00 MB** | `python hel1os_2026Oct01T154051316.py` | ~ 15 seconds |
| `results/kaggle_export/` | **8.84 MB** | `python scripts/prepare_kaggle_dataset.py` | < 1 second |
| `results/deep_learning_baseline/*` | **0.26 MB** | `python scripts/train_dnn_baseline.py` | ~ 45 seconds |
| `results/models/1d_cnn_baseline_fold_*` | **0.91 MB** | `python scripts/train_cv.py --model cnn_baseline` | ~ 90 seconds |
| `__pycache__/` | **0.22 MB** | Automatic upon running any `.py` script | Instant |

---

## 2. Detailed Regeneration Procedures

### A. Regenerating ML Dataset Arrays (`data/ml/`)
If the duplicate `X_sequences_expanded.npy` or `X_sequences_v3.npy` is deleted:
```bash
# Method 1: Re-build dataset from raw PRADAN Level-1 FITS archives
python scripts/build_dataset_v3.py

# Method 2: Fast restore from active v3 files
python -c "
import shutil
shutil.copy('data/ml/X_sequences_v3.npy', 'data/ml/X_sequences_expanded.npy')
shutil.copy('data/ml/X_sequences_v3_memmap.dat', 'data/ml/X_sequences_memmap.dat')
shutil.copy('data/ml/y_labels_v3.npy', 'data/ml/y_labels_expanded.npy')
shutil.copy('data/ml/sequence_metadata_v3.csv', 'data/ml/sequence_metadata_expanded.csv')
"
```

### B. Regenerating Kaggle Upload Package (`solar_flare_dataset_v3.zip`)
To re-create the Kaggle-ready compressed upload package and `results/kaggle_export/` directory:
```bash
python scripts/prepare_kaggle_dataset.py
```
*Result:* Generates `solar_flare_dataset_v3.zip` and populates `results/kaggle_export/` with current model weights and metadata.

### C. Re-downloading Level-1 Raw PRADAN Data
If raw ZIP archives or partial downloads in `pradan1.issdc.gov.in/` are deleted:
```bash
# Option 1: Download HEL1OS archives
python hel1os_2026Oct01T154051316.py

# Option 2: Download SoLEXS archives
python solexs_2026Oct01T154108917.py

# Option 3: Standard automated PRADAN downloader script
python scripts/download_pradan.py
```

### D. Retraining Baseline Model Checkpoints
If early baseline checkpoints are required for historical benchmark audits:
```bash
# Retrain Dense Neural Network (DNN) Baseline
python scripts/train_dnn_baseline.py

# Retrain 1D CNN 5-Fold Cross-Validation Baseline
python scripts/train_cv.py --model 1d_cnn_baseline
```

---

## 3. Guarantees & Reproducibility Verification

1. **Deterministic Random Seeds:** All dataset splits and model initializations utilize fixed random seeds (`SEED = 42`). Retraining produces identical weights and metrics.
2. **Lossless Pipeline:** The raw FITS download scripts verify SHA256 file hashes against ISRO PRADAN manifests.
3. **Inference & App Safety:** The Streamlit app (`app.py`) loads directly from `results/models/best_model.pt` and `data/test_samples/`, which are strictly protected on the **KEEP** list.
