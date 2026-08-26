# HEL1OS Data Structure & Format Verification Report

**Instrument**: High Energy L1 Orbiting X-ray Spectrometer (HEL1OS) — Aditya-L1  
**Project Context**: Solar Flare Prediction & Cross-Instrument Catalog Alignment

---

## 1. Inventory & Directory Hierarchy

The HEL1OS dataset is organized under `data/hel1os/` preserving the PRADAN raw package structure:

```
data/hel1os/
├── raw/
│   └── YYYYMMDD/
│       └── N00_0000/
│           └── HLS_YYYYMMDD_HHMMSS_XXXXXsec_lev1_V111.zip
├── extracted/
│   └── YYYYMMDD/
│       └── HLS_YYYYMMDD_HHMMSS_XXXXXsec_lev1_V111/
│           ├── aux/
│           ├── czt/
│           ├── cdte/
│           └── events/
└── processed/
```

- **Raw Package Format**: Level-1 ZIP archives (`HLS_YYYYMMDD_HHMMSS_XXXXXsec_lev1_V111.zip`).
- **Cadence**: Exactly **1.0 second** time resolution.
- **Coverage**: Synchronized across key SoLEXS solar flare observation dates.

---

## 2. FITS Product & File Classification

Inside each HEL1OS Level-1 package, files are classified into four functional directories:

| Subfolder | File Name Pattern | File Type / Description | Scientific Data? |
| :--- | :--- | :--- | :--- |
| **`cdte/`** | `lightcurve_cdte1.fits` | CdTe Detector 1 Light Curves (1.8 – 90.0 keV) | **PRIMARY SCIENCE DATA** |
| **`cdte/`** | `lightcurve_cdte2.fits` | CdTe Detector 2 Light Curves (1.8 – 90.0 keV) | **PRIMARY SCIENCE DATA** |
| **`czt/`** | `lightcurve_czt1.fits` | CZT Detector 1 Light Curves (18.0 – 160.0 keV) | **PRIMARY SCIENCE DATA** |
| **`czt/`** | `lightcurve_czt2.fits` | CZT Detector 2 Light Curves (18.0 – 160.0 keV) | **PRIMARY SCIENCE DATA** |
| **`cdte/`** | `hel1os_cdte_spectra_*.fits` | Binned Energy Spectra (511 channels) | **SECONDARY SCIENCE** |
| **`czt/`** | `hel1os_czt_spectra_*.fits` | Binned Energy Spectra (341 channels) | **SECONDARY SCIENCE** |
| **`events/`** | `evt.fits` | Raw Photon Event Lists (Individual photon arrival times) | **RAW EVENT DATA** |
| **`aux/`** | `hk.fits` | Instrument Housekeeping (Temperatures, Voltages) | Auxiliary / Calibration |
| **`aux/`** | `gticdte*.fits`, `gticzt*.fits` | Good Time Intervals (`tstart`, `tstop`) | Auxiliary / Quality Filtering |
| **`aux/cztdis/`**| `czt*dispix.txt` | Disabled Pixel Maps | Calibration |

---

## 3. Scientific Observation Data Schema

The primary science data files (`lightcurve_cdte1.fits`, `lightcurve_czt1.fits`, etc.) contain Binary Table HDUs corresponding to specific X-ray energy passbands:

### Energy Passband Extensions
1. **CdTe Detectors (`CDTE1` & `CDTE2`)**:
   - `CDTE1_LC_BAND_1.80KEV_TO_90.00KEV` (**Full Wide Band**)
   - `CDTE1_LC_BAND_5.00KEV_TO_20.00KEV`
   - `CDTE1_LC_BAND_20.00KEV_TO_30.00KEV`
   - `CDTE1_LC_BAND_30.00KEV_TO_40.00KEV`
   - `CDTE1_LC_BAND_40.00KEV_TO_60.00KEV`
2. **CZT Detectors (`CZT1` & `CZT2`)**:
   - `CZT1_LC_BAND_18.00KEV_TO_160.00KEV` (**Full Wide Band**)
   - `CZT1_LC_BAND_20.00KEV_TO_40.00KEV`
   - `CZT1_LC_BAND_40.00KEV_TO_60.00KEV`
   - `CZT1_LC_BAND_60.00KEV_TO_80.00KEV`
   - `CZT1_LC_BAND_80.00KEV_TO_150.00KEV`

### Column Schema
| Column Name | Data Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `MJD` | Float64 | Days | Modified Julian Date |
| `ISOT` | String (30A) | UTC ISO 8601 | Standard UTC timestamp (`YYYY-MM-DDTHH:MM:SS.sss`) |
| `CTR` | Float64 | counts / sec | Calibrated Count Rate |
| `STAT_ERR` | Float64 | counts / sec | 1-$\sigma$ Statistical Uncertainty Error |

---

## 4. Sampling Cadence & Time Alignment Verification

- **Cadence Statistics**:
  - `min_dt` = 1.000 s
  - `mean_dt` = 1.000 s
  - `max_dt` = 1.000 s
  - `std_dt` = 0.000 s
- **POSIX Time Conversion**:
  - Timestamps are seamlessly converted to POSIX seconds (`time` = `(time_dt - 1970-01-01T00:00:00Z).total_seconds()`).
- **SoLEXS Catalog Alignment**:
  - Both SoLEXS and HEL1OS observations use standard UTC POSIX seconds. HEL1OS light curves can be directly aligned with SoLEXS flare start, peak, and end windows ($t_{\text{start, SoLEXS}} \le t_{\text{HEL1OS}} \le t_{\text{end, SoLEXS}}$).

---

## 5. Software Utilities Implemented

1. **Extraction Script**: [extract_hel1os.py](file:///home/dharani/Desktop/solar_flare/scripts/extract_hel1os.py)
   - Unzips level-1 packages into `data/hel1os/extracted/YYYYMMDD/`.
2. **Data Loader Module**: [load_hel1os.py](file:///home/dharani/Desktop/solar_flare/scripts/load_hel1os.py)
   - `load_hel1os_light_curve()`: Parses FITS extensions into pandas DataFrames.
   - `get_hel1os_observation_summary()`: Returns observation metrics and cadence details.
3. **Exploration Notebook**: [03_hel1os_exploration.ipynb](file:///home/dharani/Desktop/solar_flare/notebooks/03_hel1os_exploration.ipynb)
   - Interactive notebook demonstrating multi-band time-series visualization and cadence verification.
