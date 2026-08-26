# Research Paper Manuscript Outline

**Target Journal**: *Solar Physics* / *Astrophysical Journal Supplement Series*  
**Report Location**: [paper_outline.md](file:///home/dharani/Desktop/solar_flare/results/paper_outline.md)  

---

## 1. Abstract
- High-frequency X-ray precursor detection using Aditya-L1 HEL1OS data.
- Stratified 5-Fold CV evaluation on 732 sequence tensors `(732, 3600, 4)`.
- Key results: **TSS = 0.6731**, **ROC-AUC = 0.9054**, **Recall = 83.33%**, **Accuracy = 86.07%**.

## 2. Introduction
- Solar flare impacts on space weather and near-Earth technological infrastructure.
- The role of Aditya-L1 at Lagrange Point L1.
- Objectives and paper organization.

## 3. Related Work
- Machine learning approaches in solar flare prediction (SVM, Random Forest, 2D CNNs on magnetograms).
- Time-series light curve modeling.

## 4. Aditya-L1 Instruments & Dataset Construction
- **SoLEXS & HEL1OS Instrument Architectures**.
- **Data Ingestion Pipeline**: FITS validation, channel scaling, quality filtering ($N \ge 2,500$ samples).
- **Tensor Structure**: `(732, 3600, 4)`. Class imbalance handling ($5.42 : 1$).

## 5. Methodology
- **Pre-processing**: Channel Z-score standardization.
- **Model Architectures**:
  - 1D CNN Baseline
  - CNN + BiLSTM Hybrid
  - CNN + Attention + BiLSTM
- **Loss Function & Training**: Weighted BCE Loss with positive weight balancing.

## 6. Experimental Benchmark & Skill Score Results
- 5-Fold Stratified Cross-Validation protocol.
- Metrics comparison table (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, TSS, HSS, POD, FAR, CSI).
- Decision threshold optimization ($0.05$ to $0.95$).

## 7. Model Explainability & Discussion
- Integrated Gradients attribution across the 3,600s lookback window.
- Soft vs. Hard X-ray physical contributions.

## 8. Limitations & Future Work
- Satellite orbital lookback gaps ($N < 2,500$).
- Future integration of SDO/HMI magnetograms.

## 9. Conclusion
- Summary of Aditya-L1 machine learning benchmark achievements.
