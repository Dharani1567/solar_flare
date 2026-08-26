# Model Architecture Reconstruction Guide

**Report Location**: `results/model_reconstruction_guide.md`  

---

## 1. Primary Model Specification (CNN + BiLSTM Hybrid)

- **Input Dimension**: `(Batch, 3600, 4)` (4 X-ray detector channels: `CdTe1`, `CdTe2`, `CZT1`, `CZT2`)
- **Convolutional Feature Extractor**:
  - `Conv1D(4 -> 32, kernel_size=7, padding=3)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)`
  - `Conv1D(32 -> 64, kernel_size=5, padding=2)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` (Output: `(Batch, 900, 64)`)
- **Recurrent Temporal Encoder**:
  - `2-Layer Bidirectional LSTM(input_size=64, hidden_size=64, dropout=0.3)` (Output: `(Batch, 900, 128)`)
- **Pooling & Classification Head**:
  - Global Max Pooling over 900 time steps -> `(Batch, 128)`
  - `Linear(128 -> 64)` + `ReLU` + `Dropout(0.5)` + `Linear(64 -> 1)`
- **Total Parameters**: `198,721` parameters

---

## 2. Hyperparameters & Cross-Validation Strategy

- **Optimizer**: Adam (`lr=2e-3`, `weight_decay=1e-4`)
- **Loss Function**: `BCEWithLogitsLoss` with positive class balancing weight (`pos_weight = N_neg / N_pos = 5.42`)
- **Cross-Validation**: Stratified 5-Fold Cross-Validation (`random_state=42`)
- **Batch Size**: `32`
- **Max Epochs & Patience**: 35 Epochs, Early Stopping patience = 5
