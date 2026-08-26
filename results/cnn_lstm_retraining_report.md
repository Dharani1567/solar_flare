# 1D CNN + Bidirectional LSTM Hybrid Model Retraining Report

**Model Architecture**: 1D CNN Feature Extractor + 2-Layer Bidirectional LSTM + Dense Classifier  
**Dataset**: Updated HEL1OS 1 Hz Time-Series Tensor (`X_sequences_expanded.npy`)  
**Sample Count ($N$)**: `732` sequences (Shape: `(732, 3600, 4)`)  
**Class Distribution**: `618` Minor Flares ($y=0$), `114` Major Flares ($y=1$, M/X class)  
**Evaluation Protocol**: Stratified 5-Fold Cross-Validation  
**Loss Function**: Binary Cross-Entropy with Pos-Weight Class Imbalance Compensation (`pos_weight = 5.42`)  
**Report Location**: [cnn_lstm_retraining_report.md](file:///home/dharani/Desktop/solar_flare/cnn_lstm_retraining_report.md) / [results/cnn_lstm_retraining_report.md](file:///home/dharani/Desktop/solar_flare/results/cnn_lstm_retraining_report.md)  

---

## 1. Network Architecture Overview

```
Input Sequence Tensor (Batch, 3600, 4)
             │
             ▼
[Transpose -> (Batch, 4, 3600)]
             │
   ┌─────────┴─────────┐
   │  1D CNN Stage 1   │  Conv1d(4->32, k=7, p=3) + BatchNorm + ReLU + MaxPool1d(2)  => (Batch, 32, 1800)
   └─────────┬─────────┘
             │
   ┌─────────┴─────────┐
   │  1D CNN Stage 2   │  Conv1d(32->64, k=5, p=2) + BatchNorm + ReLU + MaxPool1d(2) => (Batch, 64, 900)
   └─────────┬─────────┘
             │
[Transpose -> (Batch, 900, 64)]
             │
   ┌─────────┴─────────┐
   │ Bi-LSTM Backbone  │  2-Layer Bidirectional LSTM (hidden_size=64, dropout=0.3)    => (Batch, 900, 128)
   └─────────┬─────────┘
             │
   ┌─────────┴─────────┐
   │ Global MaxPool 1D │  Temporal Max Pooling over 900 time steps                     => (Batch, 128)
   └─────────┬─────────┘
             │
   ┌─────────┴─────────┐
   │ Dense Classifier  │  Linear(128->64) + ReLU + Dropout(0.5) + Linear(64->1)         => Logits (Batch, 1)
   └───────────────────┘
```

---

## 2. Stratified 5-Fold Cross-Validation Results

| Fold Index | Train $N$ | Val $N$ | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Early Stop Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 585 | 147 | 0.8776 | 0.5641 | **0.9565** | 0.7097 | 0.9397 | 0.7083 | Epoch 14 |
| **Fold 2** | 585 | 147 | 0.8571 | 0.5312 | 0.7391 | 0.6182 | 0.9130 | 0.6599 | Epoch 17 |
| **Fold 3** | 586 | 146 | 0.8219 | 0.4634 | 0.8261 | 0.5938 | 0.8713 | 0.5308 | Epoch 20 |
| **Fold 4** | 586 | 146 | **0.9247** | **0.7000** | 0.9130 | **0.7925** | 0.9258 | **0.7955** | Epoch 11 |
| **Fold 5** | 586 | 146 | 0.9110 | 0.6957 | 0.7273 | 0.7111 | **0.9468** | 0.6691 | Epoch 14 |
| **Mean ± Std** | **585.6** | **146.4** | **0.8784 ± 0.0414** | **0.5909 ± 0.1042** | **0.8324 ± 0.1021** | **0.6850 ± 0.0800** | **0.9193 ± 0.0298** | **0.6727 ± 0.0957** |

---

## 3. Aggregated Out-Of-Fold (OOF) Performance ($N=732$)

Evaluating out-of-fold predictions concatenated across all 5 validation splits:

- **OOF Accuracy**: **`0.8784`** (643 / 732 correct classifications)
- **OOF Precision**: **`0.5758`** (95 true positives / 165 positive predictions)
- **OOF Recall (Sensitivity)**: **`0.8333`** (95 true positives / 114 actual major flares)
- **OOF Specificity**: **`0.8867`** (548 true negatives / 618 actual minor flares)
- **OOF F1-Score**: **`0.6810`**
- **OOF ROC-AUC**: **`0.9167`**
- **OOF PR-AUC**: **`0.6601`**

### OOF Confusion Matrix

| | Predicted Minor ($y=0$) | Predicted Major ($y=1$) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Minor Flare ($y=0$)** | **548** (True Negatives) | **70** (False Positives) | 618 |
| **Actual Major Flare ($y=1$)** | **19** (False Negatives) | **95** (True Positives) | 114 |
| **Total Predicted** | 567 | 165 | 732 |

---

## 4. Key Performance Insights & Gains

1. **Substantial Major Flare Detection Boost**: The model achieved an **OOF Recall of 83.33%**, successfully predicting **95 out of 114 major flares** up to 60 minutes prior to peak flux.
2. **Impact of Tier-1 Ingestion**: Ingesting +36 additional M/X flares from Tier-1 dates expanded major flare training samples from 78 to 114, providing the network with diverse high-flux pre-flare dynamics.
3. **Sequential Feature Extraction**: The 2-layer Bidirectional LSTM backbone captured long-range temporal trends and derivative shifts across the 3,600s window that static CNN pooling missed.
