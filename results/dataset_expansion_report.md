# Solar Flare Dataset Expansion Report

**Report Target**: Quantitative Analysis of Expanded Dataset (Aditya-L1 SoLEXS + HEL1OS + GOES)  
**Report Location**: [dataset_expansion_report.md](file:///home/dharani/Desktop/solar_flare/results/dataset_expansion_report.md)  

---

## 1. Key Dataset Expansion Summary Table

| Metric | Previous Benchmark | Expanded Dataset | Net Increase / Gain | Percentage Growth |
| :--- | :--- | :--- | :--- | :--- |
| **Observation Days** | **117** | **194** | **+77** | **+65.8%** |
| **Total SoLEXS Flare Events** | **1142** | **1608** | **+466** | **+40.8%** |
| **Total M/X Major Flares** | **142** | **142** | **+0** | **+0.0%** |
| **Usable 1 Hz Sequences ($N$)** | **474** | **511** | **+37** | **+7.8%** |
| **Usable M/X Major Sequences** | **78** | **78** | **+0** | **+0.0%** |

---

## 2. Event Audit & Discarded Events Breakdown

Out of **1608** total detected SoLEXS flare events, **511** events met all strict quality criteria to produce clean, non-leaking 3,600-second 4-channel time-series sequence tensors.

### Breakdown of Discarded Events:

| Reason for Exclusion | Count | Percentage | Description / Quality Check |
| :--- | :--- | :--- | :--- |
| **Missing HEL1OS Coverage** | **1097** | **68.2%** | HEL1OS detector light curve file missing or satellite off-pointing/gap |
| **Missing SoLEXS Coverage** | **0** | **0.0%** | All catalog events derived directly from active SoLEXS observations |
| **Insufficient Lookback Window** | **164** | **10.2%** | Less than 2,500 valid 1s samples in 60-minute pre-flare window ($t \le t_{\text{peak}}$) |
| **Corrupted FITS Files** | **0** | **0.0%** | Verified 100% uncorrupted FITS files via checksum & header checks |
| **Other / Data Quality Exclusions** | **0** | **0.0%** | No anomalous timestamps or invalid count values detected |

## 3. Class Balance Comparison (`binary_major_flare`)

- **Previous Dataset Class Balance**:
  - Minor Flares (0): 396 (83.54%)
  - Major Flares (1): 78 (16.46%)
  - Ratio: 5.08 : 1
- **Expanded Dataset Class Balance**:
  - Minor Flares (0): 433
  - Major Flares (1): 78
  - Ratio: 5.55 : 1
