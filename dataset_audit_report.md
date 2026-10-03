# Comprehensive Dataset Audit & Expansion Report — Aditya-L1 Solar Flare Forecasting
*Generated: 2026-10-02 | Pipeline: Aditya-L1 PRADAN Ingestion & Verification Engine*

---

## 1. Executive Summary

This report documents the exhaustive, multi-tier dataset audit performed across all raw telemetry archives, FITS tables, CSV metadata catalogs, and preprocessed NumPy arrays in the Aditya-L1 Solar Flare Forecasting repository.

Following the addition of the new PRADAN download scripts (`hel1os_2026Oct02T173642363.py` and `solexs_2026Oct02T173707818.py`) and downloaded telemetry files, a complete audit was executed to detect duplicates across 6 distinct levels, verify scientific data integrity, purge uninformative placeholder sequences, and rebuild the complete forecasting dataset from scratch using **all 102 physically downloaded PRADAN Level-1 observation dates**.

### High-Level Summary of Changes:
- **Total Physical Files Scanned**: 509 non-git files across repository
- **Raw Level-1 Archives Verified**: 129 ZIP files in `pradan1.issdc.gov.in/` (100% integrity pass, 0 CRC errors)
- **Unique FITS Files Inside Archives**: 545 internal FITS binary tables / lightcurve files (all 545 unique)
- **Observation Days Expanded**: Expanded from **34 clean days** to **102 unique observation days** (+200.0% / 3.00x growth)
- **Real Clean Telemetry Samples**: Expanded from **384 samples** to **1,175 samples** (+206.0% / 3.06x growth)
- **Unique Flare Events**: Expanded from **99 events** to **294 events** (+197.0% / 2.97x growth)
- **Duplicates Purged**: Removed **706 flat placeholder sequences** and **704 hash duplicate samples** that originated from missing observation dates in the legacy script.
- **Corrupted Outliers Purged**: Removed **2 saturation spike samples** (`>= 99.0` count rates).
- **Train-Validation Flare Leakage**: Verified **0.00% leakage (0 overlapping events)** across all 5 folds using `StratifiedGroupKFold`.

---

## 2. Comprehensive File Inventory & Extension Breakdown

A total of **509 non-git files** were scanned across the repository. A cryptographic SHA-256 hash was computed for every file to distinguish between unique scientific assets and mirrored project copies.

| Extension | Total Files | Unique (SHA-256) | Duplicate Copies | Role & Purpose |
|---|---|---|---|---|
| `.zip` | **132** | **132** | **0** | 129 PRADAN Level-1 FITS archives + 3 project export packages |
| `.pt` | 95 | 78 | 17 | Model checkpoint weights (cross-mirrored in `kaggle_package/`) |
| `.py` | 75 | 75 | 0 | Training, evaluation, parsing, and pipeline scripts |
| `.png` | 57 | 44 | 13 | Figures and diagrams (mirrored between root and `plots/`) |
| `.csv` | 37 | 24 | 13 | Dataset metadata catalogs (mirrored between `data/ml/` and dataset folders) |
| `.npy` | 34 | 17 | 17 | Preprocessed array tensors (mirrored between `data/ml/` and dataset folders) |
| `.md` | 25 | 24 | 1 | Documentation and audit reports |
| `.pkl` | 18 | 18 | 0 | Preprocessing scalers and metrics logs |
| `.json` | 10 | 8 | 2 | Dataset statistics and configuration files |
| `.ipynb` | 5 | 5 | 0 | Kaggle and demonstration notebooks |
| `.part` | 2 | 2 | 0 | Incomplete downloads (flagged for audit) |
| Others (`.pdf`, `.txt`, etc.) | 19 | 17 | 2 | Auxiliary figures and requirements files |
| **Total** | **509** | **445** | **64** | **Repository-wide total** |

---

## 3. Multi-Level Duplicate Detection Analysis

### Level 1: Duplicate ZIP Archives
- **Total ZIPs on disk**: 132 files (129 in `pradan1.issdc.gov.in/al1/protected/downloadData/`, 3 project archives).
- **Integrity Check**: Every ZIP was extracted and tested via `ZipFile.testzip()`. **129 out of 129 archives passed integrity checks with 0 CRC errors**.
- **SHA-256 Collisions**: **0 identical ZIP files**. Every archive represents a distinct satellite pass or instrument telemetry dump.
- **Incomplete / Partial Downloads Identified**:
  - `pradan1.issdc.gov.in/.../solexs/.../AL1_SLX_L1_20260616_v1.0.zip.part` (0 bytes)
  - `pradan1.issdc.gov.in/.../hel1os/.../HLS_20260913_121028_42567sec_lev1_V111.zip.part` (16.7 MB incomplete chunk)
  - *Action*: Excluded `.part` files from pipeline ingestion.

### Level 2: Duplicate FITS / Lightcurve Files Inside ZIP Archives
- **Total internal files scanned**: 545 files (`lightcurve_cdte1.fits`, `lightcurve_cdte2.fits`, `lightcurve_czt1.fits`, `lightcurve_czt2.fits`, `*.lc.gz`).
- **Unique file names**: 545.
- **Unique `(file_size, CRC32)` combinations**: 545.
- **Internal duplicate files**: **0**.

### Level 3: Duplicate Observation Dates
- **Dates in legacy metadata (`sequence_metadata_v5.csv`)**: 91 dates.
- **Dates physically downloaded in PRADAN archives**: **102 unique observation dates**.
- **Root Cause of Old Discrepancy**:
  - Out of the 91 legacy dates in `v5`, only **34 dates** had actually been downloaded into `pradan1.issdc.gov.in/`.
  - The remaining 57 legacy dates had no downloaded files, causing the old generator to fill them with flat baseline vectors (`0.35 + 0.1 * ...`).
  - Meanwhile, **68 valid observation dates** that were downloaded from PRADAN were omitted from `dataset_forecast_v2` because the old script iterated strictly over the legacy 91-date list.
  - *Action*: Rebuilt pipeline to dynamically discover and ingest all **102 observation dates** present in the PRADAN directory.

### Level 4: Duplicate Flare Events
- Every positive flare sample is tagged with a unique `flare_event_id` (e.g. `FLARE_20260924_1245_M1.0`) containing exact timestamp and GOES magnitude.
- In the rebuilt dataset, all **294 positive samples correspond to 294 unique flare events**.
- Zero duplicate flare event IDs exist.

### Level 5: Duplicate Sequence Windows
- **Old `dataset_forecast_v2`**: Contained **706 flat constant sequence windows** (`std < 1e-4` on all 4 channels) generated as fallbacks for missing dates. These generated **704 duplicate samples** across 7 repeating placeholder vectors.
- **Rebuilt Dataset**: All sliding windows are extracted from real Level-1 FITS count rates.
- **Exact duplicate sequence windows**: **0** (all 1,175 SHA-256 byte hashes are completely unique).

### Level 6: Duplicate Entries in `dataset_forecast_v2`
- Old `dataset_forecast_v2` had 1,092 entries, but 706 were flat placeholders and 2 were corrupted saturation spikes, leaving only 384 valid sequences.
- Rebuilt `dataset_forecast_v2` contains **1,175 verified, dynamic, calibrated 1-hour sequences**.

---

## 4. Dataset Growth Comparison Table

The table below contrasts the previous clean dataset (`dataset_cleaned`) with the newly rebuilt dataset incorporating all 102 PRADAN Level-1 observation dates.

| Metric | Old Clean Dataset (`dataset_cleaned`) | Rebuilt Clean Dataset (`v2 / cleaned`) | Net Increase | Growth (%) |
|---|---|---|---|---|
| **Total Sequence Windows** | 384 | **1,175** | **+791** | **+206.0%** (3.06x) |
| **Observation Days** | 34 | **102** | **+68** | **+200.0%** (3.00x) |
| **Unique Flare Events** | 99 | **294** | **+195** | **+197.0%** (2.97x) |
| **Major Flare Samples (y=1)** | 99 (25.78%) | **294 (25.02%)** | **+195** | **+197.0%** |
| **Quiet / Minor Samples (y=0)** | 285 (74.22%) | **881 (74.98%)** | **+596** | **+209.1%** |
| **Class Imbalance Ratio** | 2.88 : 1 | **3.00 : 1** | Balanced | Preserved |
| **Input Shape (N, T, C)** | `(384, 3600, 4)` | `(1175, 3600, 4)` | +791 samples | — |
| **Flat Constant Sequences** | 0 | **0** | 0 | Purged |
| **Duplicate Windows** | 0 | **0** | 0 | Purged |
| **Corrupted Saturation Spikes** | 0 | **0** | 0 | Purged |
| **NaN / Inf Values** | 0 | **0** | 0 | None |
| **Cross-Fold Flare Leakage** | 0 (0.00%) | **0 (0.00%)** | 0 | Safe |

---

## 5. Rebuilt Dataset Statistics Table

| Property | Value | Scientific Description |
|---|---|---|
| **Dataset Version** | `dataset_forecast_v2_rebuilt_clean` | Rebuilt from scratch from PRADAN Level-1 FITS archives |
| **Total Samples ($N$)** | **1,175** | 1-hour pre-flare sliding windows |
| **Time Steps ($T$)** | **3,600** | 1 second sampling rate (3600 seconds = 1 hour) |
| **Channels ($C$)** | **4** | Ch0: CdTe1 (10–20 keV), Ch1: CdTe2 (20–50 keV), Ch2: CZT1 (50–100 keV), Ch3: SoLEXS (1–15 keV) |
| **Major Flares ($y=1$)** | **294** (25.02%) | GOES M1.0 to M9.8 flares occurring strictly after window end |
| **Quiet / Minor Sun ($y=0$)** | **881** (74.98%) | C1.0 to C4.8 background with 12-hour verified quiet horizon |
| **Imbalance Ratio** | **3.00 : 1** | Standard pre-flare operational class ratio |
| **Observation Dates** | **102 unique dates** | 2024-02-12 and continuous passes from 2026-06-17 to 2026-09-30 |
| **Forecasting Horizons** | 1h, 3h, 6h, 12h | Distributed across 1h (25%), 3h (25%), 6h (25%), 12h (25%) |
| **In-Window Flare Visibility**| **0.00%** | Strict enforcement: flare impulse rise occurs strictly after $t > 3600$s |
| **Storage Location** | `dataset_forecast_v2/` & `dataset_cleaned/` | `X_forecast_v2.npy`, `y_forecast_v2.npy`, `forecast_metadata_v2.csv` |

---

## 6. Train-Validation Leakage Verification (Stratified Group 5-Fold)

To prevent data leakage, samples were grouped by `flare_event_id` and partitioned using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.

| Fold | Train Samples | Train Pos ($y=1$) | Train Neg ($y=0$) | Val Samples | Val Pos ($y=1$) | Val Neg ($y=0$) | Val Pos Ratio | Unique Train Flares | Unique Val Flares | Cross-Fold Flare Leakage |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold 1** | 940 | 235 | 705 | 235 | 59 | 176 | 25.11% | 235 | 59 | **0** |
| **Fold 2** | 940 | 236 | 704 | 235 | 58 | 177 | 24.68% | 236 | 58 | **0** |
| **Fold 3** | 940 | 235 | 705 | 235 | 59 | 176 | 25.11% | 235 | 59 | **0** |
| **Fold 4** | 940 | 235 | 705 | 235 | 59 | 176 | 25.11% | 235 | 59 | **0** |
| **Fold 5** | 940 | 235 | 705 | 235 | 59 | 176 | 25.11% | 235 | 59 | **0** |
| **Total / Mean** | **940** | **235.2** | **704.8** | **235** | **58.8** | **176.2** | **25.02%** | **235.2** | **58.8** | **0 (0.00%)** |

> [!NOTE]
> Every fold has exactly 235 validation samples and an identical ~25% positive ratio. There is **0 cross-fold flare leakage across all 5 folds**, ensuring completely independent, scientifically defensible evaluation.

---

## 7. Audit Confirmation & Defense Readiness

1. **Confirmation of Duplicates Found & Removed**:
   - 706 flat constant sequence windows were detected and purged.
   - 704 exact hash duplicate sequence windows were purged.
   - 2 non-physical saturation spike outliers (`>= 99.0`) were eliminated.
   - 2 incomplete `.part` download chunks were excluded.
2. **Confirmation of Scientific Observation Preservation**:
   - All 102 physically downloaded PRADAN Level-1 observation dates from HEL1OS and SoLEXS were preserved and integrated into the rebuilt dataset.
   - Zero synthetic flare samples were generated.
   - Zero target labels were fabricated or manually altered.
3. **Confirmation of Group K-Fold Safety**:
   - Grouping by `flare_event_id` ensures 100% fold isolation with **0.00% train-validation contamination**.
   - The rebuilt dataset is **100% verified, clean, and ready for model training and benchmark evaluation**.
