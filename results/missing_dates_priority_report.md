# HEL1OS Missing Observation Dates Priority & Scientific Importance Report

**Report Location**: [missing_dates_priority_report.md](file:///home/dharani/Desktop/solar_flare/results/missing_dates_priority_report.md)  
**Analysis Timestamp**: 2026-08-26 14:56:00 IST  
**Objective**: Categorize and rank all observation dates missing HEL1OS coverage, prioritizing M/X major flare recovery.

---

## Executive Summary

Out of 194 total catalog observation dates, **94 dates** are currently extracted on disk, while **133 observation dates** (spanning **952 missing flare events**) remain to be ingested.
Crucially, **19 missing observation dates** contain **45 M/X major flares** (7 X-class flares and 38 M-class flares).

| Priority Tier | Target Criteria | Unique Dates | M/X Flares | Total Missing Flares | Script Ready on Disk | Action Required |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1 (High)** | **Contains M/X Class Flares** | **19** | **45** | **252** | **9 / 19 dates** | Download & extract Tier 1 immediately |
| **Tier 2 (Medium)** | **High Volume (>=5) or Script Available** | **100** | **0** | **661** | **80 / 100 dates** | Download script-ready Tier 2 dates |
| **Tier 3 (Low)** | **Minor / Low Volume (1-4 flares)** | **14** | **0** | **39** | **0 / 14 dates** | Query PRADAN for backfill |
| **Total** | | **133** | **45** | **952** | **89 / 133 dates** | |

---

## 1. Tier 1: High Priority Observation Dates (19 Dates with M/X Flares)

These **19 observation dates** contain all **45 missing M/X major flares** in the catalog. Recovering these dates is the highest priority to boost M/X flare sample size ($N$) for model training.

| Rank | Date | M/X Count | X Flares | M Flares | Specific M/X Flare Classes | Total Flares | PRADAN Script Status |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| 1 | `20240514` | **6** | `4` | `2` | `X1.7, X1.7, X1.2, X8.7, M4.4, M4.4` | 16 | ✓ **Ready in raw_downloads** |
| 2 | `20240523` | **6** | `0` | `6` | `M4.2, M1.7, M1.0, M1.0, M2.5, M1.0` | 38 | ✓ **Ready in raw_downloads** |
| 3 | `20240519` | **4** | `0` | `4` | `M1.9, M2.5, M1.6, M1.6` | 11 | ✓ **Ready in raw_downloads** |
| 4 | `20260603` | **3** | `1` | `2` | `M9.3, M7.7, X1.0` | 14 | ✗ Requires PRADAN Portal Fetch |
| 5 | `20260602` | **3** | `0` | `3` | `M1.2, M1.2, M3.3` | 17 | ✗ Requires PRADAN Portal Fetch |
| 6 | `20240524` | **3** | `0` | `3` | `M1.4, M1.0, M1.4` | 19 | ✓ **Ready in raw_downloads** |
| 7 | `20260516` | **3** | `0` | `3` | `M1.9, M1.3, M1.9` | 10 | ✗ Requires PRADAN Portal Fetch |
| 8 | `20240522` | **3** | `0` | `3` | `M1.5, M2.3, M1.2` | 13 | ✓ **Ready in raw_downloads** |
| 9 | `20240515` | **2** | `2` | `0` | `X3.4, X2.9` | 20 | ✓ **Ready in raw_downloads** |
| 10 | `20260620` | **2** | `0` | `2` | `M1.3, M1.0` | 9 | ✗ Requires PRADAN Portal Fetch |
| 11 | `20260621` | **2** | `0` | `2` | `M2.6, M6.8` | 10 | ✗ Requires PRADAN Portal Fetch |
| 12 | `20240517` | **1** | `0` | `1` | `M7.2` | 17 | ✓ **Ready in raw_downloads** |
| 13 | `20240521` | **1** | `0` | `1` | `M1.9` | 14 | ✓ **Ready in raw_downloads** |
| 14 | `20260510` | **1** | `0` | `1` | `M5.7` | 13 | ✗ Requires PRADAN Portal Fetch |
| 15 | `20260529` | **1** | `0` | `1` | `M1.1` | 10 | ✗ Requires PRADAN Portal Fetch |
| 16 | `20240516` | **1** | `0` | `1` | `M1.0` | 7 | ✓ **Ready in raw_downloads** |
| 17 | `20260507` | **1** | `0` | `1` | `M2.6` | 8 | ✗ Requires PRADAN Portal Fetch |
| 18 | `20260606` | **1** | `0` | `1` | `M1.8` | 5 | ✗ Requires PRADAN Portal Fetch |
| 19 | `20260517` | **1** | `0` | `1` | `M1.4` | 1 | ✗ Requires PRADAN Portal Fetch |

### Tier 1 Breakdown by Script Readiness:

- **Tier 1A (Script Ready on Disk - 9 dates)**: Dates `20240514`, `20240523`, `20240519`, `20240524`, `20240522`, `20240515`, `20240517`, `20240521`, `20240516` contain **27 M/X flares** (6 X-class, 21 M-class) and can be downloaded/extracted immediately using existing URLs in `data/hel1os/raw_downloads/`.
- **Tier 1B (Requires ISSDC Portal Fetch - 10 dates)**: Dates `20260603`, `20260602`, `20260516`, `20260620`, `20260621`, `20260510`, `20260529`, `20260507`, `20260606`, `20260517` contain **18 M/X flares** (1 X-class, 17 M-class) and require fetching fresh script links from the ISSDC PRADAN portal.

---

## 2. Tier 2: Medium Priority Observation Dates (100 Dates)

These dates contain C-class and B-class minor flares or unmatched flares. **80 of these dates** already have download scripts present in `data/hel1os/raw_downloads/`.

Top 15 Tier 2 Dates by Flare Density:

| Rank | Date | Total Flares | C Flares | B Flares | Unmatched | PRADAN Script Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `20260523` | **17** | 16 | 0 | 1 | Requires PRADAN Portal Fetch |
| 2 | `20260607` | **14** | 14 | 0 | 0 | Requires PRADAN Portal Fetch |
| 3 | `20260514` | **13** | 13 | 0 | 0 | Requires PRADAN Portal Fetch |
| 4 | `20260509` | **12** | 12 | 0 | 0 | Requires PRADAN Portal Fetch |
| 5 | `20260508` | **10** | 10 | 0 | 0 | Requires PRADAN Portal Fetch |
| 6 | `20260515` | **12** | 7 | 4 | 1 | Requires PRADAN Portal Fetch |
| 7 | `20260512` | **8** | 8 | 0 | 0 | Requires PRADAN Portal Fetch |
| 8 | `20260513` | **8** | 8 | 0 | 0 | Requires PRADAN Portal Fetch |
| 9 | `20260611` | **8** | 8 | 0 | 0 | Requires PRADAN Portal Fetch |
| 10 | `20240518` | **10** | 6 | 0 | 4 | ✓ **Ready in raw_downloads** |
| 11 | `20260519` | **9** | 6 | 2 | 1 | Requires PRADAN Portal Fetch |
| 12 | `20260526` | **7** | 7 | 0 | 0 | Requires PRADAN Portal Fetch |
| 13 | `20260605` | **7** | 7 | 0 | 0 | Requires PRADAN Portal Fetch |
| 14 | `20260608` | **7** | 7 | 0 | 0 | Requires PRADAN Portal Fetch |
| 15 | `20240508` | **20** | 0 | 0 | 20 | ✓ **Ready in raw_downloads** |

---

## 3. Tier 3: Low Priority Observation Dates (14 Dates)

Low flare count (1–4 minor flares per day) with zero M/X major flares.

---

## 4. Recommended Action Plan for Ingestion

1. **Phase 1 (Immediate Tier 1A Execution)**: Run download and extraction on the 9 Tier 1 dates with scripts already on disk (`20240514`, `20240523`, `20240519`, `20240524`, `20240522`, `20240515`, `20240517`, `20240521`, `20240516`). Recover **27 M/X flares** (6 X-class, 21 M-class).
2. **Phase 2 (ISSDC Portal Fetch for Tier 1B)**: Request download links for the remaining 10 Tier 1 dates (`20260603`, `20260602`, `20260516`, `20260620`, `20260621`, `20260510`, `20260529`, `20260507`, `20260606`, `20260517`) to recover the remaining **18 M/X flares**.
3. **Phase 3 (Sequence Re-building)**: Execute `scripts/build_sequence_dataset_expanded.py` after each phase to expand `X_sequences_expanded.npy` toward $N \approx 1,200+$ sequences.
