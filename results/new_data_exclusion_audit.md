# HEL1OS Sequence Exclusion Forensic Audit Report

**Report Location**: [new_data_exclusion_audit.md](file:///home/dharani/Desktop/solar_flare/new_data_exclusion_audit.md) / [results/new_data_exclusion_audit.md](file:///home/dharani/Desktop/solar_flare/results/new_data_exclusion_audit.md)  
**Audit Timestamp**: 2026-08-26 18:18 IST  
**Scope**: Forensic Investigation of the 14 HEL1OS ZIP Archives & Target Dates (`20240607`, `20240612`, `20240527`, `20240613`, `20240618`, `20240621`)  
**Audit Objective**: Determine exactly why dataset tensor shape remained at `(732, 3600, 4)` and major flare count remained at `114`  

---

## Executive Summary

A forensic audit was performed to determine why processing the 14 HEL1OS ZIP archives yielded **0 new sequence samples**, maintaining the dataset shape at **`(732, 3600, 4)`** and major flares at **`114`**.

**Forensic Audit Findings**:
1. **Duplicate Data & Prior Extraction**: All 6 observation dates represented in the 14 ZIP archives (`20240607`, `20240612`, `20240527`, `20240613`, `20240618`, `20240621`) were **already fully extracted** in `data/hel1os/extracted/` with **252 total FITS files** prior to this run.
2. **Non-Catalog Dates**: 5 of the 6 dates (`20240607`, `20240612`, `20240613`, `20240618`, `20240621`) are satellite background observation dates containing **0 flare events in the flare catalog**. The sequence builder only generates tensors for cataloged flares.
3. **Pre-Existing Sequence Processing**: Date `20240527` contains 13 catalog flares, of which **1 sequence sample was already incorporated in the ML dataset**. The remaining 12 events were excluded due to pre-flare satellite orbital gaps ($N_{\text{samples}} < 2,500$ in the 60-minute window).

---

## 1. Itemized Date-by-Date Forensic Inspection Table

| Target Date String | Source ZIP Count | FITS Files Extracted | Catalog Flare Count | M/X Flare Count | ML Sequences in Dataset | Sequence Builder Scanned? | Primary Forensic Reason |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `20240607` | 1 ZIP | 70 FITS | **0** | 0 | 0 | Yes | Duplicate date; 0 flares in catalog |
| `20240612` | 1 ZIP | 70 FITS | **0** | 0 | 0 | Yes | Duplicate date; 0 flares in catalog |
| `20240527` | 1 ZIP | 28 FITS | **13** | 1 (`X2.8`) | 1 | Yes | Already processed; 12 events filtered |
| `20240613` | 2 ZIPs | 28 FITS | **0** | 0 | 0 | Yes | Duplicate date; 0 flares in catalog |
| `20240618` | 4 ZIPs | 42 FITS | **0** | 0 | 0 | Yes | Duplicate date; 0 flares in catalog |
| `20240621` | 5 ZIPs | 14 FITS | **0** | 0 | 0 | Yes | Duplicate date; 0 flares in catalog |
| **Total** | **14 ZIPs** | **252 FITS** | **13 Flares** | **1 M/X** | **1 Sequence** | **100% Scanned** | **Zero New Sequences Obtainable** |

---

## 2. Event-by-Event Forensic Classification (`20240527` Catalog Flares)

Date `20240527` is the only date among the 6 ZIP target dates containing catalog flare events. All 13 flare events were audited against `build_sequence_dataset_expanded.py` quality criteria:

| Flare Event Index | Peak Time (Epoch Sec) | GOES Class | Forensic Classification | Audit Details / Exclusion Cause |
| :---: | :---: | :---: | :--- | :--- |
| **1** | `1716769147.0` | UNMATCHED | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **2** | `1716772927.0` | C3.8 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **3** | `1716774570.0` | C4.4 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **4** | `1716777546.0` | UNMATCHED | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **5** | `1716779560.0` | C6.9 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **6** | `1716781179.0` | C6.9 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **7** | `1716783890.0` | C5.5 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **8** | `1716785078.0` | C5.5 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **9** | `1716788551.0` | C2.8 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **10** | `1716793677.0` | `X2.8` | **Filtered by Builder** | `X2.8` flare peak occurred during orbital lookback gap ($N_{\text{samples}} < 2,500$) |
| **11** | `1716800707.0` | UNMATCHED | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **12** | `1716811773.0` | C6.7 | **Filtered by Builder** | Lookback sample count $< 2,500$ (Satellite orbital gap) |
| **13** | `1716840762.0` | UNMATCHED | **Sequence Generated** | **Already in Dataset** (Row in `sequence_metadata_expanded.csv`) |

---

## 3. Sequence Balance Comparison & Maximum Obtainable Yield

| Sequence Metric | Expected | Generated | Lost | Maximum Obtainable |
| :--- | :---: | :---: | :---: | :---: |
| **New Observation Dates Ingested** | 0 new | 0 | 0 | **0 dates** |
| **New Flare Events Added** | 0 new | 0 | 0 | **0 flares** |
| **Sequence Tensors ($N$)** | 0 new | 0 | 0 | **0 sequences** |
| **Major Flares ($y=1$, M/X)** | 0 new | 0 | 0 | **0 major flares** |

---

## 4. Direct Answers to Audit Questions

1. **Did Data Already Exist?**  
   **YES.** All 6 observation dates (`20240607`, `20240612`, `20240527`, `20240613`, `20240618`, `20240621`) were already extracted in `data/hel1os/extracted/` with 252 FITS files prior to processing these archives.
2. **Did Downloads Fail?**  
   **NO.** The 14 ZIP archives were present locally.
3. **Did Extraction Fail?**  
   **NO.** FITS files were already extracted and verified on disk.
4. **Were Files Duplicates?**  
   **YES.** The 14 ZIP archives contained duplicate files for observation dates already present on disk.
5. **Did Dataset Builder Exclude Them?**  
   **YES.** For `20240527`, 12 events were excluded due to pre-flare orbital gaps ($N_{\text{samples}} < 2,500$), and 5 dates contained 0 catalog flares.
