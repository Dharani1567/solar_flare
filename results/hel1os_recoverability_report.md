# HEL1OS Excluded Flare Events & Recoverability Analysis Report

**Analysis Scope**: Audit of all **1,097 flare events** excluded from 1 Hz multi-channel sequence tensor dataset due to missing HEL1OS coverage.  
**Report Location**: [hel1os_recoverability_report.md](file:///home/dharani/Desktop/solar_flare/results/hel1os_recoverability_report.md)  
**Full Excluded Events CSV**: [hel1os_excluded_events_analysis.csv](file:///home/dharani/Desktop/solar_flare/results/hel1os_excluded_events_analysis.csv)  

---

## 1. Executive Summary & Recoverability Breakdown

Out of **1,608 total detected SoLEXS flares**, **511 events** were clean and usable ($N=511$).  
The remaining **1,097 events** were excluded for the following specific reasons:

| Exclusion Category | Event Count | % of Excluded | % of Total Catalog | Primary Cause |
| :--- | :--- | :--- | :--- | :--- |
| **No Extracted HEL1OS Files on Disk** | **1,065** | **97.1%** | **66.2%** | HEL1OS `.fits` files not yet downloaded or extracted for that observation date |
| **Insufficient Pre-flare Lookback Window** | **32** | **2.9%** | **2.0%** | Satellite off-pointing or telemetry gaps during 60-min pre-flare window (<2,500s valid samples) |
| **Total Excluded Flare Events** | **1,097** | **100.0%** | **68.2%** | |

---

## 2. PRADAN Availability & Recoverability Status

Cross-referencing the **1,065 events missing from disk** against PRADAN download scripts in `data/hel1os/raw_downloads/*.py`:

| Recoverability Status | Event Count | Unique Dates | Major Flares (M/X) | Minor Flares (C/B) | Action Required |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High (Script Exists on PRADAN)** | **786** | **113** | **46** | **740** | Run `python scripts/download_hel1os.py` & `extract_hel1os.py` |
| **Medium (Query PRADAN Portal)** | **279** | **49** | **18** | **261** | Fetch download scripts for remaining 49 dates from ISSDC PRADAN |
| **Unrecoverable (Orbit Gaps)** | **32** | **21** | **2** | **30** | Permanent telemetry gap during pre-flare window |
| **Total Excluded Events** | **1,097** | **183** | **66** | **1,031** | |

### Missing Detector Channels & Files
- For all **1,065 file-missing events**, **all 4 detector channels** (`cdte1`, `cdte2`, `czt1`, `czt2`) are currently missing on local disk.
- Each observation date requires downloading 4 channel `.fits` files under `data/hel1os/raw/YYYYMMDD/`.

---

## 3. Usable Sequence Recovery Estimation

Assuming a historical **95% sequence completion rate** (accounting for minor 5% lookback window gaps):

1. **Immediate Recovery (Phase 1 - Run Existing PRADAN Scripts)**:
   - **Target Dates**: 113 dates ready in `data/hel1os/raw_downloads/*.py`
   - **Estimated New Recovered Sequences**: **~747 usable 1 Hz sequence tensors**
   - **Major Flare (M/X) Gain**: **+44 Major Sequences** (increasing M/X sequence count from 78 to **122**)
   - **New Total Dataset Size ($N$)**: Grows from **511 $\to$ 1,258 sequences** (**+146% dataset expansion**).

2. **Complete Recovery (Phase 2 - Full PRADAN Ingestion)**:
   - **Target Dates**: All 162 missing observation dates
   - **Estimated New Recovered Sequences**: **~1,010 usable 1 Hz sequence tensors**
   - **Major Flare (M/X) Gain**: **+61 Major Sequences** (increasing M/X sequence count from 78 to **139**)
   - **Final Potential Dataset Size ($N$)**: Grows from **511 $\to$ 1,521 sequences** (**+197% dataset expansion**).

---

## 4. Sample Excluded Events Audit Table (First 25 Events)

Below is a detailed sample of excluded events. The complete list of 1,097 events is stored in [hel1os_excluded_events_analysis.csv](file:///home/dharani/Desktop/solar_flare/results/hel1os_excluded_events_analysis.csv).

| Date | Peak Time (sec) | GOES Class | Missing Reason | Missing Channels | On PRADAN? | Recoverability Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20240514` | `1715651272.0` | **X1.7** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715652543.0` | **X1.7** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715657286.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715659716.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715662451.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715674530.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715681343.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715691322.0` | **X1.2** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715700469.0` | **C9.6** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715702857.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715705372.0` | **X8.7** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715708242.0` | **M4.4** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715709119.0` | **M4.4** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715714674.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715716904.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240514` | `1715728279.0` | **C7.2** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715732330.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715736471.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715738118.0` | **UNMATCHED** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715747125.0` | **C5.5** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715748885.0` | **C5.5** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715753892.0` | **C5.7** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715754285.0` | **C5.7** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715758812.0` | **C9.9** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |
| `20240515` | `1715761008.0` | **C9.9** | No extracted HEL1OS files on disk for date | `cdte1, cdte2, czt1, czt2` | `Yes` | High (Download script exists on PRADAN) |

---

## 5. Next Steps & Recommendations

1. **Trigger Automated HEL1OS Download**:
   ```bash
   python scripts/download_hel1os.py
   python scripts/extract_hel1os.py
   python scripts/build_sequence_dataset_expanded.py
   ```
2. **Re-train ML & Hybrid Deep Learning Models**:
   - Re-train Random Forest, XGBoost, and 1D CNN + LSTM hybrid models on the expanded $N \approx 1,258$ dataset to boost M/X flare detection sensitivity.
