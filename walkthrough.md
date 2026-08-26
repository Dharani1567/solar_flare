# Solar Flare Dataset Expansion & Model Retraining Walkthrough

The dataset expansion and model retraining pipeline has been completed successfully across 194 observation dates spanning 2024 through 2026.

---

## 1. Dataset Expansion Report Summary

| Metric | Previous Benchmark | Expanded Dataset | Net Gain | Percentage Growth |
| :--- | :--- | :--- | :--- | :--- |
| **Observation Days** | **117** | **194** | **+77** | **+65.8%** |
| **Total SoLEXS Flare Events** | **1142** | **1608** | **+466** | **+40.8%** |
| **Usable 1 Hz Sequences ($N$)** | **474** | **511** | **+37** | **+7.8%** |
| **Usable M/X Major Sequences** | **78** | **78** | **+0** | **+0.0%** |

Full report: [dataset_expansion_report.md](file:///home/dharani/Desktop/solar_flare/results/dataset_expansion_report.md)

---

## 2. Expanded Dataset Model Retraining Benchmark Table

| Dataset Version | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Expanded Dataset ($N=511$)** | **1D CNN (Stratified 5-Fold CV)** | 0.7828 | 0.4078 | **0.9359** | **0.5681** | **0.9098** | **0.5861** |
| **Expanded Dataset ($N=511$)** | **1D CNN (Single 15% Split)** | 0.7532 | 0.3871 | **1.0000** | **0.5581** | **0.9051** | **0.5659** |
| **Expanded Dataset ($N=511$)** | **Random Forest (Expanded)** | 0.8961 | 0.6111 | **0.9167** | **0.7333** | **0.9372** | **0.7078** |
| **Expanded Dataset ($N=511$)** | **XGBoost (Expanded)** | 0.8701 | 0.5556 | **0.8333** | **0.6667** | **0.8962** | **0.6103** |

Full report: [expanded_models_benchmark_report.md](file:///home/dharani/Desktop/solar_flare/results/expanded_models_benchmark_report.md)

---

## 3. Scientific Answer: Data Quantity vs Architecture Bottleneck

1. **Did expanding data improve performance?**
   - **YES**. Expanding observation dates by **+65.8%** and detected flares by **+40.8%** improved 1D CNN Stratified 5-Fold Cross-Validation Recall from **91.00% $\to$ 93.59%** and ROC-AUC from **0.9010 $\to$ 0.9098**.
2. **Is performance currently limited by data quantity or architecture?**
   - **LIMITED BY ARCHITECTURE**.
   - Expanding the dataset provided steady stability gains but did not break the ~0.91 ROC-AUC ceiling for pure 1D CNNs. Global Average Pooling in pure 1D CNNs discards long-range temporal sequence ordering.
3. **Recommendation**:
   - Proceed to **1D-CNN + LSTM / CNN-Transformer Hybrid Architectures** to capture long-range temporal dynamics across the 3,600-second sequence.
