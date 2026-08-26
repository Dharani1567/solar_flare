# HEL1OS + SoLEXS + GOES Flare Prediction Dataset Report

**Dataset Path**: [flare_prediction_dataset.csv](file:///home/dharani/Desktop/solar_flare/data/ml/flare_prediction_dataset.csv)  
**Total Samples (Flares)**: 1,142  
**Total Features**: 112 HEL1OS temporal features + 14 metadata/SoLEXS features  

---

## 1. Target Class Balance (`binary_major_flare`)

| Target Class | Description | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0 (Minor Flares)** | B Class, C Class, & UNMATCHED Flares | **1,000** | **87.57%** |
| **1 (Major Flares)** | M Class & X Class Flares | **142** | **12.43%** |
| **Total** | All Validated Catalog Flares | **1,142** | **100.00%** |

*Class Imbalance Insight*: The dataset exhibits a realistic ~7:1 class imbalance (12.43% positive class), ideal for evaluating Precision-Recall AUC, ROC-AUC, and F1-score in machine learning models.

---

## 2. Missing Value Analysis

- Total HEL1OS Feature Columns: **112**
- Features with Available Data: **112** / 112
- Missing Value Strategy: Missing values occur when HEL1OS data is unavailable during specific historical lookback windows (e.g. night side / orbit gaps). Standard tree-based models (XGBoost, LightGBM, Random Forest) natively handle missing values (`NaN`), while linear/neural models can use median/mean imputation.

---

## 3. Sample Feature Distributions

| Feature Name | Mean | Std | Min | Median | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hls_cdte1_60m_mean` | 15.2590 | 52.3351 | 0.1681 | 1.5242 | 553.8031 |
| `hls_cdte1_60m_median` | 0.5188 | 10.9517 | 0.0000 | 0.0000 | 252.5000 |
| `hls_cdte1_60m_max` | 335.2533 | 883.1489 | 3.0000 | 96.0000 | 8829.0000 |
| `hls_cdte1_60m_min` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `hls_cdte1_60m_std` | 43.8729 | 137.3265 | 0.5922 | 7.3419 | 1279.3895 |
| `hls_cdte1_60m_slope` | -0.0245 | 0.4586 | -10.0000 | 0.0004 | 0.5326 |
| `hls_cdte1_60m_delta` | 66.4709 | 360.1651 | -1618.0000 | 0.0000 | 4389.0000 |
| `hls_cdte1_30m_mean` | 14.9602 | 51.4667 | 0.0000 | 1.7789 | 723.9539 |
| `hls_cdte1_30m_median` | 0.3762 | 4.8738 | 0.0000 | 0.0000 | 99.5000 |
| `hls_cdte1_30m_max` | 243.0675 | 616.8525 | 0.0000 | 81.0000 | 8829.0000 |
| `hls_cdte1_30m_min` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `hls_cdte1_30m_std` | 37.4886 | 119.5907 | 0.0000 | 7.6893 | 1412.4315 |
| `hls_cdte1_30m_slope` | -0.0035 | 0.4758 | -10.0000 | 0.0015 | 1.9123 |
| `hls_cdte1_30m_delta` | 65.0826 | 362.2441 | -1912.0000 | 0.0000 | 4389.0000 |
| `hls_cdte1_15m_mean` | 24.0959 | 99.2908 | 0.0000 | 2.4244 | 1447.3056 |
| `hls_cdte1_15m_median` | 6.4972 | 47.0375 | 0.0000 | 0.0000 | 579.5000 |
| `hls_cdte1_15m_max` | 220.3861 | 601.5627 | 0.0000 | 77.0000 | 8829.0000 |
| `hls_cdte1_15m_min` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `hls_cdte1_15m_std` | 43.2893 | 151.7223 | 0.0000 | 8.8261 | 1903.7130 |
| `hls_cdte1_15m_slope` | 0.0514 | 0.6180 | -10.0000 | 0.0044 | 5.6533 |


---

## 4. Machine Learning Readiness Verification Checklist

- [x] **Sample Alignment**: Exactly 1,142 flare rows matching `flare_events_labeled.csv`.
- [x] **Target Label Defined**: `binary_major_flare` (0 = Minor B/C/UNMATCHED, 1 = Major M/X).
- [x] **Temporal Safety**: Features computed strictly from historical lookback windows ($-60	ext{m}, -30	ext{m}, -15	ext{m}, -5	ext{m}$) prior to flare peak $t_{	ext{peak}}$. Zero data leakage from post-peak observations.
- [x] **Preserved Metadata**: `date`, `source_file`, `duration_sec`, `peak_counts`, `snr`, `goes_class`, `match_time_difference_sec` intact.
- [x] **Multi-Detector Coverage**: CdTe (1.8 - 90 keV) and CZT (18 - 160 keV) wide-band count rate statistics.
