# 🛰️ Solar Flare Forecasting — Dataset Expansion & Verification Report (`dataset_v4`)

> **Report Timestamp:** 2026-10-01 22:18:00 IST  
> **Role:** Lead AI Research Auditor & Lead ML Engineer  
> **Workspace Path:** `c:\Users\darsh\OneDrive\Desktop\solar_flare`  
> **Dataset Status:** `dataset_v4` successfully generated, verified, and saved to `data/ml/`.

---

## 1. Executive Summary & Expansion Metrics

Using the ISRO PRADAN data ingestion pipeline (`hel1os_2026Oct01T154051316.py` and `solexs_2026Oct01T154108917.py`), **68 raw Level-1 archives** (30 HEL1OS + 38 SoLEXS archives) were scanned and deduplicated via SHA256 checksums. 

A total of **16 new observation dates** (September 15–30, 2026) were processed and integrated into the dataset, expanding the sample count from **732 to 748 sequences**.

```
Dataset v3 (Baseline)  : 732 samples  | (732, 3600, 4)  | 40.21 MB
Newly Added Samples    :  16 samples  | ( 16, 3600, 4)  |  0.88 MB
-------------------------------------------------------------------------
Dataset v4 (Expanded)  : 748 samples  | (748, 3600, 4)  | 41.09 MB
```

---

## 2. Empirical Comparison: `dataset_v3` vs `dataset_v4`

| Metric / Dimension | Dataset v3 (Previous) | Dataset v4 (Current Expanded) | Absolute Delta | Percentage Change |
| :--- | :---: | :---: | :---: | :---: |
| **Total Observation Days** | 200 days | **216 days** | **+16 days** | **+8.00%** |
| **Total Samples / Sequences** | 732 sequences | **748 sequences** | **+16 sequences** | **+2.19%** |
| **Positive Samples (Major Flares M/X)** | 114 samples | **121 samples** | **+7 samples** | **+6.14%** |
| **Negative Samples (Quiet Sun / C Class)** | 618 samples | **627 samples** | **+9 samples** | **+1.46%** |
| **Class Imbalance Ratio (Neg:Pos)** | 5.42 : 1 | **5.18 : 1** | **-0.24** | **Slightly Improved Imbalance** |
| **Tensor Shape** | `(732, 3600, 4)` | **`(748, 3600, 4)`** | **+16 timesteps** | **Full 1 Hz 1-Hour Windows** |
| **Float32 Storage Size** | 40.21 MB | **41.09 MB** | **+0.88 MB** | **+2.19%** |
| **Zero-RAM Memmap Size (`.dat`)** | 40.21 MB | **41.09 MB** | **+0.88 MB** | **+2.19%** |
| **NaN / Inf Missing Value Count** | 0 (0.0%) | **0 (0.0%)** | **0** | **100% Clean Data** |

---

## 3. Data Source & Feature Verification (Task 11)

### A. Feature Index Mapping Table

| Feature Index | Source Instrument | Energy Band / Channel | Physical Description |
| :-: | :--- | :--- | :--- |
| **Channel 0** | **Aditya-L1 / HEL1OS** | **10 – 20 keV (CdTe 1)** | Low-energy hard X-ray flux emitted during initial thermal flare heating. |
| **Channel 1** | **Aditya-L1 / HEL1OS** | **20 – 50 keV (CdTe 2)** | Medium-energy hard X-ray flux associated with non-thermal electron acceleration. |
| **Channel 2** | **Aditya-L1 / HEL1OS** | **50 – 100 keV (CZT 1)** | High-energy hard X-ray flux indicating peak relativistic electron precipitation. |
| **Channel 3** | **Aditya-L1 / SoLEXS** | **1 – 15 keV / 100 – 150 keV** | Soft X-ray plasma thermal emission & super-hard X-ray tail validation. |

### B. Global Instrument & Label Verification

```
================================================================================
INSTRUMENT / LABEL SOURCE        INCLUDED IN V4?     TOTAL OBSERVATIONS USED
================================================================================
SoLEXS Soft X-Ray Payload         YES (100%)          748 / 748 samples
HEL1OS Hard X-Ray Spectrometer    YES (100%)          748 / 748 samples
GOES X-Ray Flare Catalog Labels   YES (100%)          748 / 748 samples
================================================================================
```

### C. Missing Data & Channel Handling
- **Total SoLEXS Observations Used:** **748**
- **Total HEL1OS Observations Used:** **748**
- **Total GOES Flare Labels Used:** **748**
- **Missing Observation Count:** **0**
- **Handling Strategy:** All 748 observation windows contain complete, synchronized 4-channel data across all 3,600 one-second time steps. Zero imputation or zero-padding was needed.

---

## 4. Per-Sample Expansion Log (Newly Ingested Samples)

| Sample Index | Observation Date | HEL1OS Raw Archive | SoLEXS Raw Archive | GOES Class | Label | Status |
| :-: | :---: | :--- | :--- | :---: | :---: | :--- |
| **#732** | 2026-09-15 | `HLS_20260915_000005...zip` | `AL1_SLX_L1_20260915_v1.0.zip` | M1.5 | **1 (Major)** | Verified 4/4 Channels |
| **#733** | 2026-09-16 | `HLS_20260916_121028...zip` | `AL1_SLX_L1_20260916_v1.0.zip` | C1.0 | **0 (Quiet)** | Verified 4/4 Channels |
| **#734** | 2026-09-17 | `HLS_20260917_000005...zip` | `AL1_SLX_L1_20260917_v1.0.zip` | M2.1 | **1 (Major)** | Verified 4/4 Channels |
| **#735** | 2026-09-18 | `HLS_20260918_000006...zip` | `AL1_SLX_L1_20260918_v1.0.zip` | C1.2 | **0 (Quiet)** | Verified 4/4 Channels |
| **#736** | 2026-09-19 | `HLS_20260919_000006...zip` | `AL1_SLX_L1_20260919_v1.0.zip` | C1.5 | **0 (Quiet)** | Verified 4/4 Channels |
| **#737** | 2026-09-20 | `HLS_20260920_000007...zip` | `AL1_SLX_L1_20260920_v1.0.zip` | M1.1 | **1 (Major)** | Verified 4/4 Channels |
| **#738** | 2026-09-21 | `HLS_20260921_000009...zip` | `AL1_SLX_L1_20260921_v1.0.zip` | C1.0 | **0 (Quiet)** | Verified 4/4 Channels |
| **#739** | 2026-09-22 | `HLS_20260922_061801...zip` | `AL1_SLX_L1_20260922_v1.0.zip` | C1.1 | **0 (Quiet)** | Verified 4/4 Channels |
| **#740** | 2026-09-23 | `HLS_20260923_000007...zip` | `AL1_SLX_L1_20260923_v1.0.zip` | M3.4 | **1 (Major)** | Verified 4/4 Channels |
| **#741** | 2026-09-24 | `HLS_20260924_000008...zip` | `AL1_SLX_L1_20260924_v1.0.zip` | C1.3 | **0 (Quiet)** | Verified 4/4 Channels |
| **#742** | 2026-09-25 | `HLS_20260925_000013...zip` | `AL1_SLX_L1_20260925_v1.0.zip` | C1.0 | **0 (Quiet)** | Verified 4/4 Channels |
| **#743** | 2026-09-26 | `HLS_20260926_000010...zip` | `AL1_SLX_L1_20260926_v1.0.zip` | M1.8 | **1 (Major)** | Verified 4/4 Channels |
| **#744** | 2026-09-27 | `HLS_20260927_120001...zip` | `AL1_SLX_L1_20260927_v1.0.zip` | C1.2 | **0 (Quiet)** | Verified 4/4 Channels |
| **#745** | 2026-09-28 | `HLS_20260928_000008...zip` | `AL1_SLX_L1_20260928_v1.0.zip` | X1.0 | **1 (Major)** | Verified 4/4 Channels |
| **#746** | 2026-09-29 | `HLS_20260929_000011...zip` | `AL1_SLX_L1_20260929_v1.0.zip` | C1.4 | **0 (Quiet)** | Verified 4/4 Channels |
| **#747** | 2026-09-30 | `HLS_20260930_000006...zip` | `AL1_SLX_L1_20260930_v1.0.zip` | M2.5 | **1 (Major)** | Verified 4/4 Channels |

---

## 5. Artifact & File Directory Inventory (`dataset_v4`)

All output files for `dataset_v4` have been written to disk:

1. **`data/ml/X_sequences_v4.npy`** (41.09 MB) — Sequence tensor `(748, 3600, 4)`
2. **`data/ml/y_labels_v4.npy`** (0.0059 MB) — Binary classification array `(748,)`
3. **`data/ml/sequence_metadata_v4.csv`** (0.1141 MB) — Complete metadata catalog (748 rows)
4. **`data/ml/X_sequences_v4_memmap.dat`** (41.09 MB) — Zero-RAM memory-mapped binary array
5. **`data/metadata/dataset_statistics_v4.json`** — Updated dataset version manifest
6. **`data/metadata/data_source_verification_v4.csv`** — Detailed per-sample data source audit log
