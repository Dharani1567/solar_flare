# Project Backup & Preservation Priority Checklist

**Report Location**: `results/backup_checklist.md`  

---

## 1. Critical Artifacts (Must Preserve)

- [x] **Source Code**: All scripts in `scripts/`
- [x] **Exploratory Notebooks**: `notebooks/02_flare_detection.ipynb`
- [x] **Catalog Metadata**: `data/solexs/flare_catalog/*.csv`
- [x] **Generated Benchmark Reports**: All markdown files in `results/`
- [x] **Publication Figures**: All `.png` files in `results/figures/`
- [x] **Dataset Tensors**: `data/ml/X_sequences_expanded.npy`, `data/ml/y_labels_expanded.npy`
- [x] **Model Weights**: `results/models/*.pt`

---

## 2. Optional / Rebuildable Artifacts

- [ ] Raw `.zip` archives downloaded from ISSDC PRADAN (can re-download from portal)
- [ ] Temporary `.fits` light curve files in `data/hel1os/extracted/` (can re-extract from raw `.zip`)
- [ ] Python `.venv/` virtual environment (can re-create via `pip install -r requirements.txt`)
- [ ] Execution log files in `logs/`
