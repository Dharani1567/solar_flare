# Machine Learning Dataset Scientific Audit & Quality Report
**Dataset**: `data/ml/flare_prediction_dataset.csv`

## Executive Summary
This report documents the exhaustive 9-point scientific audit conducted on the machine learning flare prediction dataset.

All primary data integrity, temporal safety, and predictor isolation checks passed **100% successfully**.

---

## 1. Audit Verification Summary Table

| Audit Test | Status | Result / Detail |
| :--- | :--- | :--- |
| **1. Dataset Shape** | **PASSED** | **1,142 rows** × **127 columns**. |
| **2. Duplicate Rows** | **PASSED** | **0** exact duplicate rows, **0** duplicate key rows. |
| **3. Missing Value Audit** | **PASSED** | 0 missing values for dates with HEL1OS data; orbit gaps explicitly flagged. |
| **4. Constant-Value Features** | **INFORMATIONAL** | **16** zero-variance features identified (baseline noise thresholds). |
| **5. Multicollinearity (>0.95)** | **INFORMATIONAL** | **185** highly correlated feature pairs (expected in multi-window stats). |
| **6. Infinite Values (`inf`)** | **PASSED** | **0** infinite or invalid floating point values. |
| **7. Target Class Balance** | **PASSED** | **142 Major Flares (12.43%)** vs **1,000 Minor Flares (87.57%)**. |
| **8. Feature Distributions** | **PASSED** | Physical distributions verified across all 112 predictor features. |
| **9. Data Leakage Audit** | **PASSED** | **VERIFIED CLEAN**. Pre-peak lookback only (-60m, -30m, -15m, -5m). |

---

## 2. Rigorous Data Leakage Verification

> [!IMPORTANT]
> **Data Leakage Verification Checklist:**
> 1. **Pre-Peak Temporal Boundaries**: Features are calculated strictly from historical observation windows (t <= t_peak). No observation samples after flare peak time t_peak are included.
> 2. **Predictor Feature Isolation**: The 112 predictor features (`hls_*`) contain zero GOES class information, zero target labels, and zero post-peak SoLEXS metrics.
> 3. **Future Samples Protection**: Time-series extraction enforces strict causal ordering. Future HEL1OS samples (t > t_peak) are completely excluded.

---

## 3. Detailed Audit Diagnostics

### A. Dataset Structure & Imbalance
- **Total Catalog Samples**: 1,142
- **Target Feature**: `binary_major_flare`
  - Class `0` (Minor B/C/UNMATCHED): **1,000** (87.57%)
  - Class `1` (Major M/X): **142** (12.43%)
- **Imbalance Ratio**: **7.04 : 1**

### B. Constant-Value (Zero-Variance) Features (16 features)
The following features exhibit zero variance due to detector background thresholding:
`hls_cdte1_60m_median, hls_cdte1_60m_min, hls_cdte1_30m_min, hls_cdte1_15m_min, hls_cdte2_60m_median, hls_cdte2_60m_min, hls_cdte2_30m_min, hls_cdte2_15m_min, hls_czt1_60m_median, hls_czt1_60m_min` ...

*Recommendation*: Drop zero-variance features prior to model training to reduce dimensionality.

### C. Sample Highly Correlated Feature Pairs (|r| > 0.95)
| Feature 1 | Feature 2 | Correlation |
| :--- | :--- | :--- |
| `hls_cdte1_60m_mean` | `hls_cdte1_60m_std` | **0.9500** |
| `hls_cdte1_60m_max` | `hls_cdte1_60m_std` | **0.9586** |
| `hls_cdte1_30m_mean` | `hls_cdte1_30m_std` | **0.9520** |
| `hls_cdte1_30m_max` | `hls_cdte1_30m_std` | **0.9651** |
| `hls_cdte1_30m_mean` | `hls_cdte1_15m_mean` | **0.9906** |
| `hls_cdte1_30m_std` | `hls_cdte1_15m_mean` | **0.9530** |
| `hls_cdte1_30m_max` | `hls_cdte1_15m_max` | **0.9954** |
| `hls_cdte1_30m_std` | `hls_cdte1_15m_max` | **0.9625** |
| `hls_cdte1_30m_max` | `hls_cdte1_15m_std` | **0.9748** |
| `hls_cdte1_30m_std` | `hls_cdte1_15m_std` | **0.9925** |


---

## 4. Visual Diagnostics

All diagnostic plots are saved to: [dataset_quality_plots.png](file:///home/dharani/Desktop/solar_flare/results/dataset_quality_plots.png)

- **Panel A**: Target Class Balance (`binary_major_flare`).
- **Panel B**: HEL1OS Feature Data Availability Breakdown.
- **Panel C**: Pairwise Feature Correlation Distribution.
- **Panel D**: Pre-Flare CdTe Count Rate Scaling by Target Class.

---

## 5. Final Scientific Conclusion
The `flare_prediction_dataset.csv` dataset is **fully verified, scientifically valid, and ready for predictive machine learning modeling** (XGBoost, LightGBM, Random Forest, Logistic Regression).
