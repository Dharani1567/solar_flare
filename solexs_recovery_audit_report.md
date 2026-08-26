# SoLEXS Coverage & Recovery Audit Report

**Project**: Aditya-L1 SoLEXS & HEL1OS Solar Flare Prediction  
**Report Location**: [solexs_recovery_audit_report.md](file:///home/dharani/Desktop/solar_flare/solexs_recovery_audit_report.md) / [results/solexs_recovery_audit_report.md](file:///home/dharani/Desktop/solar_flare/results/solexs_recovery_audit_report.md)  
**Execution Timestamp**: 2026-08-26 17:16 IST  
**Audit Scope**: Full scanning of `data/solexs/flare_catalog/`, `data/solexs/extracted/`, and `data/solexs/raw_downloads/`  
**Mode**: Audit Only (Zero Downloads Triggered)  

---

## Executive Summary

A comprehensive data audit was conducted on all solar flare events, extracted files, and download manifests for the **Solar Low Energy X-ray Spectrometer (SoLEXS)** onboard Aditya-L1.

**Key Finding**: **SoLEXS data coverage is 100% COMPLETE across the entire flare catalog.** 

All **1,608 catalog flare events** across **194 unique observation dates** already have fully extracted level-1 observation files present in `data/solexs/extracted/` (which contains **436 total extracted observation dates**). **Zero catalog dates are missing SoLEXS coverage.**

Consequently, **SoLEXS is NOT the bottleneck** for dataset expansion or model training. The actual bottleneck limiting 4-channel sequence generation is **HEL1OS hard X-ray light curve coverage**, where 119 catalog observation dates remain to be ingested.

---

## 1. SoLEXS Catalog & Extracted Inventory Audit

| Audit Category | Metric Value | Coverage Percentage | Audit Notes |
| :--- | :---: | :---: | :--- |
| **Total Catalog Flare Events** | **1,608 events** | 100.0% | `flare_events_labeled_expanded.csv` |
| **Total Catalog Observation Dates** | **194 dates** | 100.0% | Unique dates in expanded catalog |
| **Total Extracted SoLEXS Dates on Disk** | **436 dates** | -- | `data/solexs/extracted/` |
| **Catalog Dates with Extracted SoLEXS** | **194 dates** | **100.0%** | All 194 catalog dates extracted |
| **Catalog Dates Missing SoLEXS Data** | **0 dates** | **0.0%** | Zero missing dates |
| **Catalog Flares with Valid SoLEXS Coverage** | **1,608 events** | **100.0%** | 1,608 / 1,608 flares covered |
| **Missing SoLEXS Flare Events** | **0 events** | **0.0%** | Zero missing events |
| **M/X Major Flares in Catalog** | **142 flares** | 100.0% | 135 M-class, 7 X-class |
| **M/X Flares with Valid SoLEXS Data** | **142 flares** | **100.0%** | All 142 M/X flares covered |

---

## 2. Cross-Reference against Download Manifests & Scripts

- **Raw Download Scripts Inspected**: 4 scripts in `data/solexs/raw_downloads/`:
  - `solexs_2026Aug25T033517387.py`
  - `solexs_2026Aug25T030020352.py`
  - `solexs_2026Aug25T000217623.py`
  - `solexs_2026Aug24T234549473.py`
- **Total Unique Dates Covered by Manifests**: **433 observation dates**.
- **Missing Catalog Dates in Manifests**: **0 dates**.
- **Recoverable / Non-Recoverable SoLEXS Dates**:
  - **Recoverable Dates**: **0 dates** (100% already extracted on disk).
  - **Non-Recoverable Dates**: **0 dates**.

---

## 3. Current vs. Potential ML Dataset Calculations

| Metric Component | Current ML Dataset | Potential ML Dataset | Net Increase Potential | Primary Bottleneck |
| :--- | :---: | :---: | :---: | :--- |
| **SoLEXS Covered Flare Events** | **1,608 flares** | **1,608 flares** | +0 flares | None (100% Covered) |
| **HEL1OS Covered Flare Events** | **871 flares** | **1,608 flares** | +737 flares | **HEL1OS Ingestion** |
| **Usable 4-Channel Sequences ($N$)** | **732 sequences** | **~1,200 - 1,350** | **+468 to +618** | **HEL1OS Satellite Gaps** |
| **Major Flare Sequences ($y=1$, M/X)** | **114 sequences** | **~140 - 145** | **+26 to +31** | **HEL1OS Satellite Gaps** |
| **Minor Flare Sequences ($y=0$, C/B/U)** | **618 sequences** | **~1,060 - 1,205** | **+442 to +587** | **HEL1OS Satellite Gaps** |

---

## 4. Storage Footprint & Requirement Analysis

| Metric | SoLEXS Measurement | HEL1OS Measurement | Comparison & Takeaway |
| :--- | :---: | :---: | :--- |
| **Total Extracted Storage on Disk** | **3.97 GB** (436 dates) | **77.21 GB** (105 dates) | HEL1OS consumes ~20x more storage |
| **Average Footprint per Day** | **9.33 MB / day** | **1,271.0 MB / day** | SoLEXS footprint is extremely lightweight |
| **Storage Needed for SoLEXS Recovery** | **0.00 MB** | -- | SoLEXS recovery is 100% complete |
| **Current Free Disk Space** | **7.44 GB** | **7.44 GB** | Available on `/` (`/dev/nvme0n1p5`) |
| **Safe Processing Batch Size** | **N/A** (0 MB needed) | **5 dates / batch** | ~6.2 GB per HEL1OS batch |

---

## 5. Itemized Summary & Lists

### Missing SoLEXS Dates List:
- **`NONE`** (All 194 catalog dates are extracted in `data/solexs/extracted/`).

### Missing SoLEXS Flare Events List:
- **`NONE`** (All 1,608 catalog flare events have valid SoLEXS light curve files).

### Recoverable vs Non-Recoverable SoLEXS Dates:
- **Recoverable SoLEXS Dates**: 0 dates (Fully satisfied).
- **Non-Recoverable SoLEXS Dates**: 0 dates.

---

## 6. Final Strategic Recommendations

1. **Is SoLEXS now the bottleneck?**
   - **NO.** SoLEXS has **100% catalog coverage** across all 1,608 flare events and 194 observation dates. SoLEXS storage is minimal (9.33 MB/day).

2. **How many additional samples can be gained?**
   - **Zero additional samples needed from SoLEXS.**
   - Up to **+468 to +618 additional 4-channel sequences** can be gained by completing **HEL1OS data ingestion** across the remaining 119 HEL1OS dates.

3. **Which dates should be downloaded first?**
   - Focus 100% of download efforts on the **remaining 5 HEL1OS Tier-1B dates**:
     1. `20260510` (`M5.7` flare)
     2. `20260529` (`M1.1` flare)
     3. `20260507` (`M2.6` flare)
     4. `20260606` (`M1.8` flare)
     5. `20260517` (`M1.4` flare)
   - Following Tier-1B HEL1OS completion, execute **HEL1OS Tier-2 batch ingestion** in storage-safe 5-date batches (~6.2 GB per batch).
