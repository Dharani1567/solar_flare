# Tier-1B Recovery Forensic Audit Report

**Report Location**: [tier1b_forensic_audit.md](file:///home/dharani/Desktop/solar_flare/tier1b_forensic_audit.md) / [results/tier1b_forensic_audit.md](file:///home/dharani/Desktop/solar_flare/results/tier1b_forensic_audit.md)  
**Audit Timestamp**: 2026-08-26 17:31 IST  
**Target Scope**: 5 Tier-1B HEL1OS Observation Dates (`20260510`, `20260529`, `20260507`, `20260606`, `20260517`)  
**Audit Mode**: Audit Only (Zero Downloads Triggered)  

---

## Executive Summary

A forensic audit was conducted to investigate why the sequence dataset size ($N = 732$) and major flare count ($y=1$, 114 flares) remained unchanged following the execution of the Tier-1B recovery pipeline for dates `20260510`, `20260529`, `20260507`, `20260606`, and `20260517`.

**Core Finding**: The 5 target dates **were not downloaded** because download URL manifests for these specific 2026 dates are **absent from the local raw download scripts** in `data/hel1os/raw_downloads/`. 

When the pipeline executed, empty directory placeholders were created in `data/hel1os/extracted/`, but no `.zip` files were available in `data/hel1os/raw/` to extract. Consequently, `build_sequence_dataset_expanded.py` found **0 FITS light curve files** for these 5 dates and excluded their 37 catalog flares (including 5 M-class flares: `M5.7`, `M1.1`, `M2.6`, `M1.8`, `M1.4`), preserving the dataset size at $N = 732$.

---

## 1. Itemized Date-by-Date Inspection Table

| Target Date | Folder Exists in `extracted/` | FITS File Count | Total Size on Disk | Creation Timestamp | Modification Timestamp | Raw ZIP Status | Script Manifest Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `20260510` | **TRUE** | **0** | **0.00 MB** | 2026-08-26 17:23:21 | 2026-08-26 17:23:21 | Absent in `raw/` | Absent in `raw_downloads/` |
| `20260529` | **TRUE** | **0** | **0.00 MB** | 2026-08-26 17:23:21 | 2026-08-26 17:23:21 | Absent in `raw/` | Absent in `raw_downloads/` |
| `20260507` | **TRUE** | **0** | **0.00 MB** | 2026-08-26 17:23:21 | 2026-08-26 17:23:21 | Absent in `raw/` | Absent in `raw_downloads/` |
| `20260606` | **TRUE** | **0** | **0.00 MB** | 2026-08-26 17:23:21 | 2026-08-26 17:23:21 | Absent in `raw/` | Absent in `raw_downloads/` |
| `20260517` | **TRUE** | **0** | **0.00 MB** | 2026-08-26 17:23:21 | 2026-08-26 17:23:21 | Absent in `raw/` | Absent in `raw_downloads/` |

---

## 2. Comparison of Pipeline States (Before vs. After)

| System Component | State Before Tier-1B | State After Tier-1B | Net Change | Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **Extracted Observation Folders** | 105 folders | **110 folders** | **+5 folders** | Empty directories created by `mkdir` |
| **Total Extracted FITS Files** | 4,491 files | **4,491 files** | **+0 files** | Zero FITS files extracted or added |
| **ML Dataset Sample Count ($N$)** | **732 sequences** | **732 sequences** | **+0 sequences** | Dataset builder skipped empty dates |
| **Major Flare Count ($y=1$, M/X)** | **114 flares** | **114 flares** | **+0 flares** | 5 target M-flares skipped due to 0 FITS |
| **Minor Flare Count ($y=0$, C/B/U)** | **618 flares** | **618 flares** | **+0 flares** | 32 minor flares skipped due to 0 FITS |
| **`X_sequences_expanded.npy`** | `(732, 3600, 4)` | `(732, 3600, 4)` | **0 rows added** | Retained previous tensor state |
| **Free Disk Space** | 7.44 GB | 7.44 GB | 0.00 GB | No download or storage overhead used |

---

## 3. Categorical Evaluation of Potential Causes

### 1. Did Data Already Exist?
- **NO.**
- Inspection of `data/hel1os/extracted/20260510` (and remaining 4 dates) confirms directory folders exist but contain **0 FITS files** and **0.00 MB total disk space**.

### 2. Did Downloads Fail?
- **YES (Pre-download URL Requirement Missing).**
- Local download scripts (`hel1os_2026Aug25T033548292.py`, `hel1os_2026Aug26T084448022.py`) do not contain download URLs for these 2026 observation dates.
- Querying `https://pradan1.issdc.gov.in` directly confirmed that ISSDC PRADAN portal requires an active Keycloak SSO session cookie from an interactive user login to generate download links for these specific dates.

### 3. Did Extraction Fail?
- **NO.**
- Zero `.zip` files were present in `data/hel1os/raw/YYYYMMDD/`, so extraction was never initiated.

### 4. Were Files Duplicates?
- **NO.**
- Zero files were downloaded, extracted, or duplicated.

### 5. Did Dataset Builder Exclude Them?
- **YES.**
- `build_sequence_dataset_expanded.py` scanned `data/hel1os/extracted/`, found 0 light curve FITS files for these 5 dates (`matching_files = []`), and skipped all 37 flare events associated with these dates during 60-minute 1 Hz sequence generation.

---

## 4. Empirical Forensic Evidence

### Evidence A: AST Parsing of Raw Download Scripts
AST inspection of all Python scripts in `data/hel1os/raw_downloads/` confirmed zero URL string matches for the 5 target dates:
```python
# AST Inspection Result across all scripts in data/hel1os/raw_downloads/
Date '20260510': 0 matching URLs found
Date '20260529': 0 matching URLs found
Date '20260507': 0 matching URLs found
Date '20260606': 0 matching URLs found
Date '20260517': 0 matching URLs found
```

### Evidence B: Directory Listing & File Count Audit
```bash
$ ls -la data/hel1os/extracted/20260510/
total 8
drwxrwxr-x 2 dharani dharani 4096 Aug 26 17:23 .
drwxrwxr-x 112 dharani dharani 4096 Aug 26 17:23 ..
# Result: 0 files, 0 FITS files, 0 bytes
```

### Evidence C: Keycloak Authentication Barrier on ISSDC Portal
Direct HTTP request to ISSDC PRADAN Level-1 directory endpoints returned Keycloak SSO sign-in HTML:
```html
<title>Sign in to Indian Space Science Data Center</title>
<!-- HTTP 200 Redirect to ISSDC SSO Keycloak Authentication Portal -->
```

---

## 5. Corrective Action Plan for True Tier-1B Ingestion

To recover the 5 missing M-class flares (`M5.7`, `M1.1`, `M2.6`, `M1.8`, `M1.4`) and expand $N$:

1. **Obtain Active PRADAN Download Script**:
   Log into `https://pradan1.issdc.gov.in`, select HEL1OS Level-1 data for dates `20260510`, `20260529`, `20260507`, `20260606`, and `20260517`, and generate a fresh download script with an active session cookie.
2. **Execute Storage-Safe Ingestion**:
   Save the generated script into `data/hel1os/raw_downloads/` and execute `scripts/execute_hel1os_download_and_extract.py`.
3. **Rebuild & Retrain**:
   Execute `scripts/build_sequence_dataset_expanded.py` to ingest the newly extracted FITS files and expand $N$ past 732 sequences.
