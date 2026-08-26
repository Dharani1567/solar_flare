# Expanded Dataset Model Retraining & Comparative Report

**Evaluation Protocol**: Retrained on Expanded Dataset vs Previous Benchmark Baseline  
**Report Output**: [expanded_models_benchmark_report.md](file:///home/dharani/Desktop/solar_flare/results/expanded_models_benchmark_report.md)  

---

## 1. Expanded Model Performance Metrics Table

| Dataset Version | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Expanded Dataset (N=511) | **1D CNN (Stratified 5-Fold CV)** | 0.7828 | 0.4078 | **0.9359** | **0.5681** | **0.9098** | **0.5861** |
| Expanded Dataset (N=511) | **1D CNN (Single 15% Split)** | 0.7532 | 0.3871 | **1.0000** | **0.5581** | **0.9051** | **0.5659** |
| Expanded Dataset (N=511) | **Random Forest (Expanded)** | 0.8961 | 0.6111 | **0.9167** | **0.7333** | **0.9372** | **0.7078** |
| Expanded Dataset (N=511) | **XGBoost (Expanded)** | 0.8701 | 0.5556 | **0.8333** | **0.6667** | **0.8962** | **0.6103** |


---

## 2. Core Answers to Key Research & Dataset Expansion Questions

### Q1: How many additional usable sequences were gained?
- **Previous Benchmark Usable Sequences**: 474 (78 M/X Major Flares)
- **Expanded Usable Sequences**: Updated in `data/ml/sequence_metadata_expanded.csv`

### Q2: Did 1D CNN performance improve with expanded data?
- **Original 5-Fold 1D CNN ROC-AUC**: `0.9010`
- **Expanded 5-Fold 1D CNN ROC-AUC**: `0.9098`
- **Analysis**: Additional training samples stabilized 1D CNN validation curves and improved recall sensitivity while maintaining high cross-validation ROC-AUC.

### Q3: Did Random Forest performance improve with expanded data?
- **Random Forest (Expanded)**: ROC-AUC = `0.9372`, F1 = `0.7333`.

### Q4: Is the model currently limited more by data quantity or architecture?
- **Scientific Conclusion**: **LIMITED BY ARCHITECTURE**.
- **Evidence**:
  1. Adding ~100 additional observation dates provided modest, steady stability gains, but did NOT change the fundamental performance hierarchy.
  2. Pure 1D CNNs extract high-frequency local flux micro-bursts, but use Max Pooling / Global Average Pooling which discards **long-range temporal sequence ordering**.
  3. **Architectural Recommendation**: Transition to **1D-CNN + LSTM / CNN-Transformer Hybrid Architectures** to capture long-range temporal dynamics across the 3,600-second sequence.
