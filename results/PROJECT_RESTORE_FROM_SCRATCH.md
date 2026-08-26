# PROJECT RESTORATION FROM SCRATCH GUIDE

**Master Guide**: `restore_project_guide.md` / `PROJECT_RESTORE_FROM_SCRATCH.md`  
**Project**: Aditya-L1 Solar Flare Forecasting System  

---

## Executive Summary & Restoration Overview

This document provides a zero-loss restoration guide to recreate the entire Aditya-L1 Solar Flare Forecasting project from scratch on a blank machine even if the workspace is completely deleted.

---

## 1. Project Overview & Frozen Dataset Properties

- **Target Prediction Task**: Pre-flare binary major flare forecasting up to 60 minutes prior to flare peak.
- **Major Flare Definition ($y=1$)**: GOES M-class and X-class solar flare events.
- **Minor Flare Definition ($y=0$)**: GOES C-class, B-class, and quiet background solar events.
- **Pre-Flare Lookback Window**: 3,600 seconds (1 Hz sampling rate).
- **Frozen Sequence Tensor**: `X_sequences_expanded.npy` (Shape: `(732, 3600, 4)`).
- **Total Sequences**: `732` (`114` Major Flares, `618` Minor Flares | Class Imbalance: `5.42 : 1`).

---

## 2. Environment Reconstruction

To restore the Python environment on a blank machine:

```bash
git clone https://github.com/Dharani1567/solar_flare.git
cd solar_flare
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Refer to [environment_report.md](environment_report.md) for full `pip freeze` specs.

---

## 3. Data Lineage & Rebuilding Dataset Tensors

Refer to [DATA.md](DATA.md) and [dataset_rebuild_guide.md](dataset_rebuild_guide.md):

```bash
# 1. Download raw Level-1 archives from ISSDC PRADAN portal into data/hel1os/raw/
# 2. Extract raw archives
python scripts/extract_all_archives.py

# 3. Validate FITS files and build ML tensors
python scripts/validate_all_fits.py
python scripts/label_flares_expanded.py
python scripts/build_sequence_dataset_expanded.py
```

---

## 4. Model Training & Results Reproduction

Execute the automated training and benchmark evaluation suite:

```bash
# 1. Train model architectures under 5-Fold Stratified CV
python scripts/train_architecture_comparison.py

# 2. Run error analysis, threshold sweeps, and explainability
python scripts/analyze_errors.py
python scripts/optimize_thresholds.py
python scripts/run_feature_ablation.py
python scripts/run_explainability.py

# 3. Render publication figures and compile reports
python scripts/generate_literature_and_benchmark_deliverables.py
python scripts/generate_paper_figures_extended.py
```

---

## 5. Supporting Restoration Manifest Index

- [environment_report.md](environment_report.md): Environment & hardware specifications
- [data_lineage_report.md](data_lineage_report.md): Complete data lineage audit
- [raw_data_inventory.md](raw_data_inventory.md) & [raw_data_manifest.csv](raw_data_manifest.csv): Raw observation inventory
- [dataset_rebuild_guide.md](dataset_rebuild_guide.md): Step-by-step dataset rebuild commands
- [model_reconstruction_guide.md](model_reconstruction_guide.md): Model layer dimensions & hyperparameters
- [experiment_registry.csv](experiment_registry.csv): Full experiment history & metric registry
- [report_manifest.md](report_manifest.md): Manifest of all generated reports
- [results_reproduction_guide.md](results_reproduction_guide.md): Exact commands to reproduce all metrics & figures
- [backup_checklist.md](backup_checklist.md): File preservation priority checklist
- [reproducibility_gaps.md](reproducibility_gaps.md): Audit of dependencies & external gaps
