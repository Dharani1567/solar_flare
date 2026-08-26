# ISSDC PRADAN HEL1OS Data Acquisition Checklist & Gap Audit Report

**Report Location**: [missing_pradan_requests.md](file:///home/dharani/Desktop/solar_flare/missing_pradan_requests.md) / [results/missing_pradan_requests.md](file:///home/dharani/Desktop/solar_flare/results/missing_pradan_requests.md)  
**Audit Timestamp**: 2026-08-26 17:34 IST  
**Audit Target**: `data/hel1os/raw_downloads/` (4 PRADAN Scripts Inspected)  
**Catalog Reference**: `data/solexs/flare_catalog/flare_events_labeled_expanded.csv` (194 Observation Dates, 1,608 Flares)  
**Audit Mode**: Audit Only (Zero Downloads Triggered, Zero Placeholder Folders Created)  

---

## Executive Summary

An audit of all 4 Python download manifests in `data/hel1os/raw_downloads/` was conducted to map HEL1OS satellite coverage against the 194 flare observation dates in the catalog.

- **Covered Catalog Dates**: **150 dates** (77.3% coverage) are present in local download scripts.
- **Missing Catalog Dates**: **44 dates** (22.7% coverage gap) are missing from all local scripts.
- **Missing Flare Events**: **284 flare events** (out of 1,608 total catalog events) reside on these 44 missing dates.
- **Missing Major Flares**: **18 M/X major flares** (1 X-class, 17 M-class flares) reside across **10 Tier-1 dates**.

---

## 1. Summary of Manifest Coverage & Gaps

| Audit Category | Count | Percentage | Description / Impact |
| :--- | :---: | :---: | :--- |
| **Total PRADAN Scripts in `raw_downloads/`** | **4 scripts** | -- | `hel1os_2026Aug25T004405012.py`, etc. |
| **Total Unique Dates Covered by Scripts** | **207 dates** | -- | Includes non-catalog background dates |
| **Catalog Dates Covered by Scripts** | **150 dates** | **77.32%** | Ready for ingestion from script manifests |
| **Catalog Dates Missing from Scripts** | **44 dates** | **22.68%** | Requires fresh ISSDC PRADAN portal fetch |
| **Total Flare Events on Missing Dates** | **284 flares** | **17.66%** | Gaps in 1 Hz sequence tensor generation |
| **Missing M/X Major Flares** | **18 flares** | **12.68%** | 1 X-class (`X1.0`), 17 M-class flares |

---

## 2. Ranked Missing Dates by Dataset Growth Impact

The 44 missing dates are categorized into 4 Scientific Priority Tiers and ranked by **Impact Score** ($S = 100 \times N_X + 20 \times N_M + 3 \times N_C + N_B + N_{\text{unmatched}}$):

### Tier 1: High Priority (10 Dates with 18 M/X Major Flares)
Recovering these 10 dates directly expands positive major flare training samples ($y=1$).

| Rank | Date String | Total Flares | M/X Count | Specific M/X Flare Classes | Minor Flares | Impact Score | PRADAN Action Required |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **1** | `20260603` | **14** | **3** | `M9.3, M7.7, X1.0` | 11 | **171** | Fetch script from PRADAN Portal |
| **2** | `20260602` | **17** | **3** | `M1.2, M1.2, M3.3` | 14 | **102** | Fetch script from PRADAN Portal |
| **3** | `20260516` | **10** | **3** | `M1.9, M1.3, M1.9` | 7 | **81** | Fetch script from PRADAN Portal |
| **4** | `20260620` | **9** | **2** | `M1.3, M1.0` | 7 | **57** | Fetch script from PRADAN Portal |
| **5** | `20260621` | **10** | **2** | `M2.6, M6.8` | 8 | **56** | Fetch script from PRADAN Portal |
| **6** | `20260510` | **13** | **1** | `M5.7` | 12 | **52** | Fetch script from PRADAN Portal |
| **7** | `20260529` | **10** | **1** | `M1.1` | 9 | **45** | Fetch script from PRADAN Portal |
| **8** | `20260507` | **8** | **1** | `M2.6` | 7 | **35** | Fetch script from PRADAN Portal |
| **9** | `20260606` | **5** | **1** | `M1.8` | 4 | **32** | Fetch script from PRADAN Portal |
| **10** | `20260517` | **1** | **1** | `M1.4` | 0 | **20** | Fetch script from PRADAN Portal |

---

### Tier 2: Medium Priority (11 High Volume Dates $\ge 8$ Flares)
Provides high-density C-class minor flare sequences ($y=0$) to expand background dataset size.

| Rank | Date String | Total Flares | C-Class Flares | B-Class Flares | Unmatched | Impact Score | PRADAN Action Required |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **11** | `20260523` | **17** | 16 | 0 | 1 | **49** | Fetch script from PRADAN Portal |
| **12** | `20260607` | **14** | 14 | 0 | 0 | **42** | Fetch script from PRADAN Portal |
| **13** | `20260514` | **13** | 13 | 0 | 0 | **39** | Fetch script from PRADAN Portal |
| **14** | `20260509` | **12** | 12 | 0 | 0 | **36** | Fetch script from PRADAN Portal |
| **15** | `20260515` | **12** | 7 | 4 | 1 | **26** | Fetch script from PRADAN Portal |
| **16** | `20260508` | **10** | 10 | 0 | 0 | **30** | Fetch script from PRADAN Portal |
| **17** | `20260519` | **9** | 6 | 2 | 1 | **21** | Fetch script from PRADAN Portal |
| **18** | `20260512` | **8** | 8 | 0 | 0 | **24** | Fetch script from PRADAN Portal |
| **19** | `20260513` | **8** | 8 | 0 | 0 | **24** | Fetch script from PRADAN Portal |
| **20** | `20260611` | **8** | 8 | 0 | 0 | **24** | Fetch script from PRADAN Portal |
| **21** | `20260518` | **10** | 6 | 0 | 4 | **22** | Fetch script from PRADAN Portal |

---

### Tier 3: Moderate Priority (14 Dates with 4–7 Flares)
Dates: `20260526` (7 flares), `20260605` (7 flares), `20260608` (7 flares), `20260522` (6 flares), `20260528` (6 flares), `20260604` (6 flares), `20260511` (5 flares), `20260520` (5 flares), `20260521` (5 flares), `20260524` (5 flares), `20260525` (5 flares), `20260527` (5 flares), `20260601` (4 flares), `20260610` (4 flares).

### Tier 4: Low Priority (9 Dates with 1–3 Flares)
Dates: `20260506` (3 flares), `20260609` (3 flares), `20260612` (3 flares), `20260613` (2 flares), `20260614` (2 flares), `20260615` (1 flare), `20260616` (1 flare), `20260617` (1 flare), `20260618` (1 flare).

---

## 3. ISSDC PRADAN Data Acquisition Checklist

Follow these steps to obtain fresh download scripts for the 44 missing observation dates:

- [ ] **Step 1**: Log into the ISSDC PRADAN Portal (`https://pradan1.issdc.gov.in`) using valid credentials.
- [ ] **Step 2**: Navigate to **Aditya-L1 Data Access** $\rightarrow$ **HEL1OS** $\rightarrow$ **Level-1 Data**.
- [ ] **Step 3 (Tier 1 Priority)**: Select the **10 Tier-1 Dates**:
  - `20260603`, `20260602`, `20260516`, `20260620`, `20260621`, `20260510`, `20260529`, `20260507`, `20260606`, `20260517`.
- [ ] **Step 4**: Click **"Generate Python Download Script"** and save as `data/hel1os/raw_downloads/hel1os_tier1_missing.py`.
- [ ] **Step 5 (Tier 2 Priority)**: Select the **11 Tier-2 High-Volume Dates** (`20260523`, `20260607`, `20260514`, etc.).
- [ ] **Step 6**: Export to `data/hel1os/raw_downloads/hel1os_tier2_missing.py`.
- [ ] **Step 7**: Verify that the generated scripts contain `url_prefix`, `cookie_string`, and `data_file_paths`.
- [ ] **Step 8**: Run storage-safe ingestion pipeline `scripts/execute_hel1os_download_and_extract.py`.
