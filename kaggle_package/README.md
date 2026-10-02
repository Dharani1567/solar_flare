# Solar Flare Forecasting — Kaggle Package

## Overview

A complete benchmarking pipeline for binary solar flare classification using real X-ray telemetry from India's **Aditya-L1** spacecraft (HEL1OS + SoLEXS instruments).

**Task**: Given a 1-hour window of multi-channel X-ray photon counts, predict whether a **major solar flare (M/X-class)** occurred within that window.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | ISRO Aditya-L1 (HEL1OS + SoLEXS) |
| **Dataset version** | v5 (latest) |
| **Tensor shape** | `(1092, 3600, 4)` — 1092 samples × 3600 sec × 4 channels |
| **Labels** | Binary — `1` = Major M/X flare, `0` = Quiet/C-class |
| **Class ratio** | 2.77:1 (negative:positive) |
| **Date range** | 2024-05-14 to 2026-09-30 (91 observation days) |

### Channels
| # | Instrument | Band |
|---|---|---|
| 0 | HEL1OS CdTe 1 | 10–20 keV |
| 1 | HEL1OS CdTe 2 | 20–50 keV |
| 2 | HEL1OS CZT 1  | 50–100 keV |
| 3 | HEL1OS CZT 2 / SoLEXS | 100–150 keV / 1–15 keV |

---

## Models Compared

| # | Model | Description |
|---|---|---|
| 1 | **1D CNN** | Baseline temporal CNN |
| 2 | **CNN + BiLSTM** | CNN feature extractor + bidirectional LSTM |
| 3 | **CNN + Attention + BiLSTM** | Attention-gated CNN-BiLSTM |
| 4 | **TCN** | Temporal Convolutional Network with dilated convolutions |
| 5 | **InceptionTime** | Multi-scale inception modules for time series |
| 6 | **ResNet1D** | Residual network adapted for 1D sequences |

---

## Package Structure

```
kaggle_package/
├── notebook.ipynb        ← Main Kaggle notebook (GPU-ready, run top-to-bottom)
├── train.py              ← Training script (all 6 models, StratifiedGroupKFold)
├── evaluate.py           ← Evaluation + plotting script
├── requirements.txt      ← Dependencies
├── README.md             ← This file
├── results/
│   └── benchmark_results.csv   ← Auto-generated after training
└── plots/
    └── *.png                   ← Auto-generated after training
```

---

## Quick Start (Kaggle)

1. Upload the dataset to Kaggle (or link via Kaggle Dataset: `aditya-l1-solar-flare-v5`)
2. Open `notebook.ipynb` in Kaggle with **GPU T4 x2** enabled
3. Run all cells from top to bottom
4. Results in `results/benchmark_results.csv`, plots in `plots/`

## Quick Start (Local)

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

---

## Evaluation Priority

Primary: **ROC-AUC** → **F1 Score**
Secondary: Precision, Recall, Accuracy

---

## Scientific Integrity Notes

- **No label manipulation** — all results are real
- **StratifiedGroupKFold** with `date` as group — prevents temporal leakage
- **Per-fold normalization** — stats computed on training fold only
- **Fixed seed 42** — fully reproducible
- **Class weighting** — `pos_weight = 2.77` for imbalanced learning
