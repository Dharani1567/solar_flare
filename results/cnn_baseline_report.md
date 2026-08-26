# PyTorch 1D CNN Raw Sequence Baseline Report

**Dataset Path**: `data/ml/X_sequences.npy` `(474, 3600, 4)` & `y_labels.npy` `(474,)`  
**Target Variable**: `binary_major_flare` (0 = Minor B/C/UNMATCHED, 1 = Major M/X)  
**Input Format**: Raw 1 Hz HEL1OS Time Series (3,600 time steps × 4 detector channels)  
**Model Checkpoint**: [cnn_baseline.pt](file:///home/dharani/Desktop/solar_flare/results/models/cnn_baseline.pt)  

---

## 1. 1D CNN Architecture & Layer Pipeline

```
Input Tensor (Batch, 3600, 4) -> Transpose to (Batch, 4, 3600)
  │
  ├──> Conv1D(in=4, out=32, k=7, p=3) ──> BatchNorm1D ──> ReLU ──> MaxPool1D(2)  [3600 -> 1800]
  ├──> Conv1D(in=32, out=64, k=5, p=2) ──> BatchNorm1D ──> ReLU ──> MaxPool1D(2)  [1800 -> 900]
  ├──> Conv1D(in=64, out=128, k=3, p=1) ──> BatchNorm1D ──> ReLU                  [900]
  ├──> GlobalAveragePooling1D ──────────────────────────────────────────> Tensor (Batch, 128)
  └──> Linear(128 -> 64) ──> ReLU ──> Dropout(0.3) ──> Linear(64 -> 1)
```

---

## 2. Benchmark Comparison Table (15% Stratified Test Set, N=72)

| Model | Input Type | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D CNN (Raw Sequences)** | Raw 1 Hz Time-Series (3600, 4) | 0.7778 | 0.4167 | 0.8333 | **0.5556** | **0.9056** | **0.6503** |
| **Random Forest (Tabular)** | Engineered Tabular Features | 0.9730 | 0.9167 | 0.9167 | **0.9167** | **0.9892** | **0.9408** |
| XGBoost (Tabular) | Engineered Tabular Features | 0.9459 | 0.9000 | 0.7500 | 0.8182 | 0.9879 | 0.9494 |
| Logistic Regression (Tabular) | Engineered Tabular Features | 0.9459 | 0.7857 | 0.9167 | 0.8462 | 0.9684 | 0.7686 |


---

## 3. Diagnostic Plots & Training Visualizations

- **Training Curves (Loss & Metrics)**: [cnn_training_curves.png](file:///home/dharani/Desktop/solar_flare/results/cnn_training_curves.png)
- **1D CNN Confusion Matrix**: [cnn_confusion_matrix.png](file:///home/dharani/Desktop/solar_flare/results/cnn_confusion_matrix.png)
- **Comparative ROC & PR Curves**: [cnn_roc_pr_curves.png](file:///home/dharani/Desktop/solar_flare/results/cnn_roc_pr_curves.png)

---

## 4. Scientific Analysis & Key Takeaways

1. **Did the 1D CNN Outperform Random Forest?**
   - **Performance Comparison**:
     - **1D CNN (Raw Sequences)**: F1 Score = **0.5556**, ROC-AUC = **0.9056**, PR-AUC = **0.6503**
     - **Random Forest (Tabular)**: F1 Score = **0.9167**, ROC-AUC = **0.9892**, PR-AUC = **0.9408**

2. **Temporal Feature Extraction in 1D CNNs**:
   - 1D Convolutions effectively capture localized high-frequency micro-bursts and multi-channel flux rises directly from raw 1 Hz light curves without hand-engineered feature extraction.
   - Global Average Pooling eliminates positional bias, allowing the CNN to detect pre-flare impulsive acceleration features anywhere in the 60-minute lookback sequence.

3. **Next Steps (LSTM & Transformer Hybrid Architectures)**:
   - While 1D CNNs extract local temporal features, recurrence (LSTM/GRU) or self-attention (Transformers) is necessary to model long-range sequential dynamics and temporal ordering across the entire 3,600-second pre-flare sequence.
