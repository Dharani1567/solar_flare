# HEL1OS Recovery Audit & Storage Management Plan

**Report Location**: [hel1os_recovery_plan.md](file:///home/dharani/Desktop/solar_flare/hel1os_recovery_plan.md) / [results/hel1os_recovery_plan.md](file:///home/dharani/Desktop/solar_flare/results/hel1os_recovery_plan.md)  
**Audit Timestamp**: 2026-08-26 17:40 IST  
**Audit Target**: `data/hel1os/extracted/`, `data/hel1os/raw_downloads/`, and `df -h` System Disk Storage  
**Audit Mode**: Audit Only (Zero Downloads Triggered, Zero Folders Created, Zero Datasets Modified)  

---

## Executive Summary

A comprehensive pre-download storage audit was conducted to verify extracted FITS inventories, raw download script availability, and system disk storage limits before executing further HEL1OS satellite data recovery.

- **System Free Disk Space (`df -h /`)**: **7.17 GB** available out of 195.80 GB (96% capacity used).
- **Extracted Observation Inventory**: **103 valid dates** extracted in `data/hel1os/extracted/` containing **4,491 total FITS files** (yielding $N = 732$ 1 Hz sequence tensors).
- **Average Extracted Storage Footprint**: **0.7496 GB (767.61 MB) per date**.
- **Truly Missing Catalog Dates**: **124 observation dates** (80 script-ready in `raw_downloads/`, 44 requiring PRADAN portal fetch).
- **Projected Storage Required for Full Recovery**: **92.95 GB** (124 dates × 0.75 GB/date).
- **Projected Free Space After Full Recovery**: **-85.78 GB** (Storage Deficit).

> [!CAUTION]
> **SAFETY STOP TRIGGERED**: Attempting to download all 124 missing observation dates at once would exhaust available disk space (-85.78 GB deficit) and crash the system. 
> **RECOMMENDED ACTION**: Execute recovery strictly in **storage-safe 5-date batches** (~3.75 GB per batch) with immediate ZIP archive deletion.

---

## 1. System Disk Space & Empirical Footprint Audit

| Storage Metric | Empirical Measurement | Source / Calculation |
| :--- | :---: | :--- |
| **Total Disk Size** | **195.80 GB** | `/dev/nvme0n1p5` filesystem |
| **Used Disk Space** | **178.62 GB** | System disk usage (96% full) |
| **Current Free Disk Space (`df -h`)** | **7.17 GB** | Live system free space |
| **Safety Cutoff Threshold** | **3.00 GB** | Minimum allowable free space |
| **Available Buffer Above Safety Threshold** | **4.17 GB** | 7.17 GB – 3.00 GB |
| **Valid Extracted Dates on Disk** | **103 dates** | `data/hel1os/extracted/` with FITS > 0 |
| **Total Extracted FITS Files** | **4,491 files** | Verified uncorrupted FITS light curves |
| **Average Extracted Footprint per Date** | **0.7496 GB (767.61 MB)** | 77.21 GB total / 103 dates |
| **Projected Storage Required (124 Missing)** | **92.95 GB** | 124 dates × 0.7496 GB/date |
| **Projected Free Space After Full Recovery** | **-85.78 GB** | **EXHAUSTED (Storage Deficit)** |
| **Recommended Safe Batch Size** | **5 dates / batch** | **~3.75 GB / batch** |

---

## 2. Comprehensive Inventory Audit Table of Catalog Dates

Below is the complete audit table of catalog observation dates, flare densities, download script readiness, FITS counts, directory sizes, and recovery status:

### Summary Breakdown by Status:
- **Already Extracted (FITS > 0)**: **103 dates** (4,491 FITS files, 77.21 GB)
- **Script Ready (Needs Ingestion)**: **80 dates** (Present in `raw_downloads/`, 0 FITS extracted yet)
- **Requires PRADAN Portal Fetch**: **44 dates** (Missing from `raw_downloads/`, 0 FITS extracted yet)
- **Total Catalog Dates**: **194 dates**

### Top 35 High-Priority Dates Gap Analysis:

| Date String | Total Flares | M/X Count | M/X Flare Classes | Script Available | FITS Count | Directory Size | Recovery Status |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `20240514` | 16 | 6 | `X1.7, X1.7, X1.2, X8.7, M4.4, M4.4` | Yes | 42 | 3,265.8 MB | **Already Extracted** |
| `20240523` | 38 | 6 | `M4.2, M1.7, M1.0, M1.0, M2.5, M1.0` | Yes | 56 | 605.1 MB | **Already Extracted** |
| `20240519` | 11 | 4 | `M1.9, M2.5, M1.6, M1.6` | Yes | 28 | 249.6 MB | **Already Extracted** |
| `20260603` | 14 | 3 | `M9.3, M7.7, X1.0` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20260602` | 17 | 3 | `M1.2, M1.2, M3.3` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20240524` | 19 | 3 | `M1.4, M1.0, M1.4` | Yes | 84 | 996.1 MB | **Already Extracted** |
| `20260516` | 10 | 3 | `M1.9, M1.3, M1.9` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20240522` | 13 | 3 | `M1.5, M2.3, M1.2` | Yes | 84 | 996.8 MB | **Already Extracted** |
| `20240515` | 20 | 2 | `X3.4, X2.9` | Yes | 56 | 3,078.5 MB | **Already Extracted** |
| `20260620` | 9 | 2 | `M1.3, M1.0` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20260621` | 10 | 2 | `M2.6, M6.8` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20240517` | 17 | 1 | `M7.2` | Yes | 42 | 583.8 MB | **Already Extracted** |
| `20240521` | 14 | 1 | `M1.9` | Yes | 98 | 1,115.4 MB | **Already Extracted** |
| `20260510` | 13 | 1 | `M5.7` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20260529` | 10 | 1 | `M1.1` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20240516` | 7 | 1 | `M1.0` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20260507` | 8 | 1 | `M2.6` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20260606` | 5 | 1 | `M1.8` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20260517` | 1 | 1 | `M1.4` | No | 0 | 0.0 MB | **Requires PRADAN Portal Fetch** |
| `20240518` | 10 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20240520` | 6 | 0 | `None` | Yes | 28 | 249.6 MB | **Already Extracted** |
| `20240525` | 12 | 0 | `None` | Yes | 84 | 996.1 MB | **Already Extracted** |
| `20240526` | 14 | 0 | `None` | Yes | 56 | 605.1 MB | **Already Extracted** |
| `20240527` | 11 | 0 | `None` | Yes | 56 | 605.1 MB | **Already Extracted** |
| `20240528` | 9 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20240530` | 8 | 0 | `None` | Yes | 56 | 605.1 MB | **Already Extracted** |
| `20240531` | 7 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20240601` | 6 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20240604` | 8 | 0 | `None` | Yes | 56 | 605.1 MB | **Already Extracted** |
| `20240605` | 7 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20240607` | 14 | 0 | `None` | Yes | 56 | 605.1 MB | **Already Extracted** |
| `20240608` | 9 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20240609` | 8 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |
| `20240610` | 10 | 0 | `None` | Yes | 56 | 605.1 MB | **Already Extracted** |
| `20240611` | 8 | 0 | `None` | Yes | 42 | 548.0 MB | **Already Extracted** |

---

## 3. Storage-Safe 5-Date Batch Execution Protocol

To safely ingest missing dates without violating the **3.0 GB free space safety threshold**:

```
[System Free Space: 7.17 GB]
             │
             ▼
[Batch 1: 5 Dates (~3.75 GB)] ──> Download & Extract ──> Auto-Delete ZIPs ──> [Free Space: ~3.42 GB]
             │
             ▼
[Storage Safety Audit Check] ──> Free Space >= 3.0 GB? ──> Continue to Next Batch
             │
             ▼
[Batch 2: 5 Dates (~3.75 GB)] ──> Requires prior disk pruning / cleaning to maintain > 3.0 GB buffer
```

### Batch Ingestion Guidelines:
1. **Batch Size Limit**: Exactly **5 dates per execution run**.
2. **Immediate Archive Cleanup**: Raw `.zip` files must be unlinked immediately after FITS extraction.
3. **Automatic Cutoff**: If system free space drops below **3.00 GB**, the execution pipeline automatically halts without modifying datasets.
