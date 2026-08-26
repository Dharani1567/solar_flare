# PyTorch Deep Neural Network (DNN) Baseline & Comparison Report

**Dataset Path**: `data/ml/flare_prediction_dataset.csv`  
**Target Variable**: `binary_major_flare` (0 = Minor B/C/UNMATCHED, 1 = Major M/X)  
**Split Protocol**: Stratified **70% Train / 15% Validation / 15% Test**  

---

## 1. PyTorch Deep Neural Network Architecture

```
Input Tensor (X_dim = 112 features)
  │
  ├──> Linear(112 -> 256) ──> BatchNorm1d ──> ReLU ──> Dropout(p=0.3)
  ├──> Linear(256 -> 128) ───────────────────> ReLU ──> Dropout(p=0.3)
  ├──> Linear(128 -> 64)  ───────────────────> ReLU
  └──> Linear(64 -> 1)    ──> BCEWithLogitsLoss (Weighted pos_weight)
```

- **Loss Function**: `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` for severe class imbalance.
- **Optimization**: Adam (`lr=1e-3`, `weight_decay=1e-4`), `ReduceLROnPlateau(patience=5)`, Early Stopping (`patience=15`).
- **Best Checkpoint Saved**: [best_dnn_model.pt](file:///home/dharani/Desktop/solar_flare/results/deep_learning_baseline/best_dnn_model.pt)

---

## 2. Model Performance Benchmark Comparison Table

Evaluated on the exact same **15% Stratified Test Set**:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch DNN** | 0.8378 | 0.5000 | 0.6667 | **0.5714** | **0.7359** | **0.6648** |
| Logistic Regression | 0.9324 | 0.7692 | 0.8333 | 0.8000 | 0.9671 | 0.7596 |
| **Random Forest** | 0.9730 | 0.9167 | 0.9167 | **0.9167** | **0.9892** | **0.9408** |
| XGBoost | 0.9459 | 0.9000 | 0.7500 | 0.8182 | 0.9879 | 0.9494 |


---

## 3. Comparative Diagnostic Visualizations

- **Training & Validation Curves**: [dnn_training_curves.png](file:///home/dharani/Desktop/solar_flare/results/deep_learning_baseline/dnn_training_curves.png)
- **PyTorch DNN Confusion Matrix**: [dnn_confusion_matrix.png](file:///home/dharani/Desktop/solar_flare/results/deep_learning_baseline/dnn_confusion_matrix.png)
- **Comparative ROC & PR Curves**: [dnn_roc_pr_curves.png](file:///home/dharani/Desktop/solar_flare/results/deep_learning_baseline/dnn_roc_pr_curves.png)

---

## 4. Rigorous Scientific Analysis & Conclusions

### A. Did the PyTorch DNN Outperform Random Forest?
**Outcome**: **NO**.
- **Random Forest**: F1 Score = **0.9167**, ROC-AUC = **0.9892**, PR-AUC = **0.9408**
- **PyTorch DNN**: F1 Score = **0.5714**, ROC-AUC = **0.7359**, PR-AUC = **0.6648**

### B. Scientific Explanation of Performance
1. **Tabular Feature Representation**: Hand-crafted summary statistics (`mean`, `max`, `std`, `slope`) over lookback windows are axis-aligned continuous features. Random Forest and gradient boosted trees excel at learning non-linear decision boundaries on tabular tabular summary features without requiring large dataset scale.
2. **Sample Size & Multi-Layer Perceptron (MLP) Bottleneck**: Deep Neural Networks with dense linear layers tend to overfit or under-express when trained on tabular aggregates of moderate size (N=492) compared to decision trees.

### C. Recommendation for Advanced Deep Learning Architecture
1. **Transition from Tabular Summary Features to Raw 1 Hz Time Series**:
   - Dense Multilayer Perceptrons (MLPs) operating on pre-aggregated tabular features cannot capture the full high-frequency micro-burst structure of solar flares.
2. **Recommended Architecture**: **1D-CNN + LSTM / Temporal Transformer Hybrid Network**
   - **Input**: Raw 1 Hz HEL1OS multi-channel light curve sequences `(batch_size, sequence_length=3600, channels=4)`.
   - **1D-CNN Frontend**: Extracts local temporal features (micro-bursts, rapid flux rises, energy spectral slope variations).
   - **LSTM / Transformer Backbone**: Models long-range temporal dependencies across the 60-minute pre-flare sequence.
