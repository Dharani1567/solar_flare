# Data Access & Reproducibility Guide (`DATA.md`)

**Mission**: Aditya-L1 — India's First Solar Observatory at Sun-Earth Lagrangian Point L1  
**Primary Payloads**: High Energy L1 Orbiting X-ray Spectrometer (**HEL1OS**) & Solar Low Energy X-ray Spectrometer (**SoLEXS**)  
**Data Repository Provider**: Indian Space Science Data Centre (**ISSDC**)  
**Data Access Portal**: [ISSDC PRADAN Portal](https://pradan.issdc.gov.in)  

---

## 1. Data Provenance & Restricted Distribution Policy

To respect ISRO data distribution policies and repository storage limits, raw FITS data files (`.fits`) and binary sequence tensors (`.npy`) are **not redistributed directly within this repository**.

Instead, this guide provides complete, step-by-step instructions to acquire the original Level-1 data archives from ISSDC PRADAN and execute our open-source ingestion and preprocessing pipeline to reproduce the exact machine learning dataset (`X_sequences_expanded.npy`, shape: `(732, 3600, 4)`).

---

## 2. Instrument & Dataset Overview

### A. HEL1OS (High Energy L1 Orbiting X-ray Spectrometer)
- **Energy Band**: 10 keV to 150 keV (Hard X-rays).
- **Detector Channels**:
  - `CdTe1` (Cadmium Telluride Spectrometer Channel 1, 10–60 keV)
  - `CdTe2` (Cadmium Telluride Spectrometer Channel 2, 10–60 keV)
  - `CZT1` (Cadmium Zinc Telluride Spectrometer Channel 1, 20–150 keV)
  - `CZT2` (Cadmium Zinc Telluride Spectrometer Channel 2, 20–150 keV)
- **Data Cadence**: High-resolution 1 Hz photon count light curves.

### B. SoLEXS (Solar Low Energy X-ray Spectrometer)
- **Energy Band**: 1 keV to 30 keV (Soft X-rays).
- **Physical Role**: Measures coronal thermal plasma heating and background solar X-ray irradiances.

### C. Dataset Statistics
- **Extracted Observation Dates**: `105` uncorrupted observation dates (`4,491` FITS light curves).
- **Sequence Count ($N$)**: `732` sequence tensors of shape `(732, 3600, 4)`.
- **Pre-Flare Lookback Window**: 3,600 seconds (60 minutes prior to flare peak).
- **Class Distribution**:
  - **Minor Flares ($y=0$, C/B/Quiet)**: `618` sequences (84.42%)
  - **Major Flares ($y=1$, M/X Class)**: `114` sequences (15.58%)

---

## 3. Step-by-Step Dataset Reproduction Workflow

### Step 1: Register on ISSDC PRADAN Portal
1. Navigate to the official ISSDC PRADAN portal: [https://pradan.issdc.gov.in](https://pradan.issdc.gov.in).
2. Register an account and sign in via Keycloak Single Sign-On (SSO).

### Step 2: Request Level-1 Observation Archives
1. Select **Aditya-L1 Mission** -> **HEL1OS** & **SoLEXS** Level-1 Science Payloads.
2. Select target observation date range (`2024-05-01` to `2024-06-30`).
3. Download the zipped Level-1 observation archives (`.zip`) into `data/hel1os/raw/` and `data/solexs/raw/`.

### Step 3: Execute Ingestion & Dataset Building Pipeline
Run the preprocessing and dataset builder scripts in sequence:

```bash
# 1. Extract raw ZIP archives and validate FITS integrity
python scripts/extract_all_archives.py
python scripts/validate_all_fits.py

# 2. Extract 4-channel 1 Hz light curve time-series and generate 60-min sequence tensors
python scripts/build_sequence_dataset_expanded.py

# 3. Train models and generate benchmark reports
python scripts/generate_literature_and_benchmark_deliverables.py
python scripts/generate_paper_figures_extended.py
```

---

## 4. Primary Data Integrity Criteria

- **Sample Count Threshold**: Sequences are generated only when the pre-flare 3,600-second lookback window contains $N \ge 2,500$ uncorrupted 1 Hz photon count samples.
- **Detector Channel Integrity**: All 4 detector channels (`CdTe1`, `CdTe2`, `CZT1`, `CZT2`) must be present and non-null.
