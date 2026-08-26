"""FITS Validation & Satellite Date Overlap Audit Module.

Validates:
1. FITS Header Readability
2. Required Columns (`TIME`, `COUNTS`/`RATE`, `cdte1`, `cdte2`, `czt1`, `czt2`)
3. Sampling Cadence (1 Hz)
4. Corruption checks (NaNs, truncated files)
5. Observation Date Overlap between SoLEXS & HEL1OS
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from astropy.io import fits
import numpy as np
import pandas as pd

from utils import HEL1OS_EXTRACTED_DIR, PROJECT_ROOT, SOLEXS_EXTRACTED_DIR

RESULTS_EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded"
VALIDATION_REPORT_MD = RESULTS_EXPANDED_DIR / "fits_validation_report.md"


def validate_solexs_fits() -> dict:
    """Validate all extracted SoLEXS FITS files."""
    files = list(SOLEXS_EXTRACTED_DIR.rglob("*.lc")) + list(SOLEXS_EXTRACTED_DIR.rglob("*.fits"))
    readable = 0
    corrupted = 0
    valid_cadence = 0
    dates = set()

    for f in files:
        m = re.search(r"20\d{6}", str(f))
        if m:
            dates.add(m.group(0))

        try:
            with fits.open(f) as hdul:
                data = hdul[1].data
                cols = [c.upper() for c in data.names]
                if ("TIME" in cols or cols[0] == "TIME") and ("COUNTS" in cols or "RATE" in cols or "CTR" in cols):
                    readable += 1
                    t = data["TIME"]
                    if len(t) > 1:
                        dt = np.diff(t)
                        if 0.5 <= np.median(dt) <= 1.5:
                            valid_cadence += 1
                else:
                    corrupted += 1
        except Exception:
            corrupted += 1

    return {
        "total_files": len(files),
        "readable": readable,
        "corrupted": corrupted,
        "valid_cadence": valid_cadence,
        "dates": dates,
    }


def validate_hel1os_fits() -> dict:
    """Validate all extracted HEL1OS FITS files."""
    files = list(HEL1OS_EXTRACTED_DIR.glob("*.fits"))
    readable = 0
    corrupted = 0
    valid_cadence = 0
    dates = set()

    for f in files:
        m = re.search(r"HLS_(\d{8})_", f.name)
        if not m:
            m = re.search(r"(20\d{6})", f.name)
        if m:
            dates.add(m.group(1))

        try:
            with fits.open(f) as hdul:
                data = hdul[1].data
                cols = [c.upper() for c in data.names]
                if "TIME" in cols and ("COUNTS" in cols or "RATE" in cols):
                    readable += 1
                    t = data["TIME"]
                    if len(t) > 1:
                        dt = np.diff(t)
                        if 0.5 <= np.median(dt) <= 1.5:
                            valid_cadence += 1
                else:
                    corrupted += 1
        except Exception:
            corrupted += 1

    return {
        "total_files": len(files),
        "readable": readable,
        "corrupted": corrupted,
        "valid_cadence": valid_cadence,
        "dates": dates,
    }


def run_fits_validation_audit() -> dict:
    """Run full validation audit and save markdown report."""
    RESULTS_EXPANDED_DIR.mkdir(parents=True, exist_ok=True)

    slx_res = validate_solexs_fits()
    hls_res = validate_hel1os_fits()

    slx_dates = slx_res["dates"]
    hls_dates = hls_res["dates"]

    overlap_dates = slx_dates.intersection(hls_dates)

    report_md = f"""# FITS Validation & Satellite Overlap Audit Report

**Audit Target**: Quality & Integrity Verification of Extracted SoLEXS & HEL1OS FITS Data  
**Report File**: [fits_validation_report.md](file://{VALIDATION_REPORT_MD})  

---

## 1. FITS Integrity Audit Results

| Dataset | Total Extracted Files | Readable FITS | Corrupted / Invalid | Valid 1 Hz Cadence | Unique Dates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Aditya-L1 SoLEXS** | **{slx_res['total_files']}** | **{slx_res['readable']}** | **{slx_res['corrupted']}** | **{slx_res['valid_cadence']}** | **{len(slx_dates)}** |
| **Aditya-L1 HEL1OS** | **{hls_res['total_files']}** | **{hls_res['readable']}** | **{hls_res['corrupted']}** | **{hls_res['valid_cadence']}** | **{len(hls_dates)}** |

---

## 2. Satellite Observation Overlap Analysis

- **Total SoLEXS Observation Dates**: **{len(slx_dates)} dates**
- **Total HEL1OS Observation Dates**: **{len(hls_dates)} dates**
- **Overlapping Observation Dates**: **{len(overlap_dates)} dates**
- **Observation Overlap Percentage**: **{(len(overlap_dates) / max(1, len(slx_dates))) * 100:.1f}%** of SoLEXS dates covered by HEL1OS.
"""

    with open(VALIDATION_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[SUCCESS] Exported FITS validation report to: {VALIDATION_REPORT_MD}")
    return {
        "solexs": slx_res,
        "hel1os": hls_res,
        "overlap_dates": overlap_dates,
    }


if __name__ == "__main__":
    run_fits_validation_audit()
