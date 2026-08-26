# HEL1OS FITS File Integrity & Detector Channel Validation Report

**Report Location**: [fits_validation_report.md](file:///home/dharani/Desktop/solar_flare/results/fits_validation_report.md)  
**Execution Timestamp**: 2026-08-26 18:15:25 IST  
**Validation Engine**: `astropy.io.fits` Header & Checksum Verification  
**Total Extracted FITS Files Audited**: `4491` FITS files  

---

## 1. FITS Integrity & Checksum Audit

- **Total FITS Files Audited**: `4491` FITS light curve files
- **Uncorrupted / Valid FITS Files**: `4491` FITS files (**100.0% Pass Rate**)
- **Corrupted / Invalid FITS Files**: `0` files (Auto-removed)

---

## 2. Detector Channel Coverage Audit

The pipeline verified the 4 required detector channels (`cdte1`, `cdte2`, `czt1`, `czt2`) across all extracted date directories in `data/hel1os/extracted/`:

| Channel Name | Detector Module | Channel Status | Primary Energy Band | Verification Check |
| :--- | :--- | :---: | :--- | :---: |
| **`cdte1`** | CdTe Spectrometer 1 | **ACTIVE** | Soft / Medium X-rays | **✓ Verified** |
| **`cdte2`** | CdTe Spectrometer 2 | **ACTIVE** | Soft / Medium X-rays | **✓ Verified** |
| **`czt1`** | CZT Spectrometer 1 | **ACTIVE** | Hard X-rays | **✓ Verified** |
| **`czt2`** | CZT Spectrometer 2 | **ACTIVE** | Hard X-rays | **✓ Verified** |

---

## 3. Light Curve Continuity & Timing Verification

- **Time Resolution**: 1.0 second (1 Hz sampling rate)
- **Quality Check**: Minimum lookback sample count (>= 2,500 out of 3,600 per 60-minute window) enforced prior to sequence tensor generation.
