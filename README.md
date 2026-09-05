# Aditya-L1 Solar Flare Forecasting — Deep Learning Suite ☀️🛰️

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Dedicated Deep Learning Codebase & Interactive Web Application**  
> *Precursor M/X Class Solar Flare Forecasting using 1 Hz Multi-Channel X-Ray Light Curve Tensors from ISRO's Aditya-L1 (SoLEXS & HEL1OS).*

---

## 🏆 Deep Learning Benchmark Overview

This repository branch (`dl-model-only`) contains the clean core deep learning architectures, trained PyTorch model checkpoints, feature interpretability pipelines, and an interactive Streamlit web dashboard.

| Model Architecture | Parameters | Key Strengths |
| :--- | :---: | :--- |
| **CNN + BiLSTM Hybrid** | `198,721` | **Top Skill Score (TSS = 0.6731)** & M/X Flare Recall (POD = 83.33%) |
| **CNN + Attention + BiLSTM** | `207,041` | **Highest Accuracy (86.07%)** & Precision (54.48%) |
| **1D CNN Baseline** | `47,425` | Ultra-fast baseline feature extractor for real-time 1 Hz streaming |
| **Temporal ConvNet (TCN)** | `154,209` | Dilated causal convolutions capturing multi-scale lookback windows |

---

## 📁 Repository Structure (`dl-model-only` Branch)

```
solar_flare/
├── app.py                                            # Interactive Streamlit Web Application
├── requirements.txt                                  # Python dependencies
├── README.md                                         # Deep Learning Branch Documentation
│
├── scripts/                                          # Core Deep Learning Execution Scripts
│   ├── train_architecture_comparison.py              # Main 5-Fold Cross-Validation trainer & evaluator
│   ├── train_cnn_5fold.py                            # 5-Fold 1D-CNN baseline model trainer
│   ├── train_cnn_lstm_expanded.py                    # Hybrid CNN-LSTM model trainer
│   ├── run_explainability.py                         # Integrated Gradients & Temporal Saliency attribution
│   ├── run_feature_ablation.py                       # Channel & temporal feature ablation experiments
│   ├── build_sequence_dataset_expanded.py            # Sequence dataset builder for PyTorch tensors
│   ├── generate_sample_test_files.py                 # Sample sequence dataset generator for testing
│   ├── test_streamlit_app.py                         # Automated tester for app & PyTorch models
│   └── utils.py                                      # Shared helper utility paths
│
├── data/
│   ├── ml/                                           # Prepared tensor datasets (X_sequences, y_labels)
│   └── test_samples/                                 # Lightweight sample .npy / .csv sequence inputs
│
└── results/
    └── models/                                       # Trained PyTorch model checkpoints (.pt)
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install requirements:

```bash
git clone -b dl-model-only https://github.com/Dharani1567/solar_flare.git
cd solar_flare
pip install -r requirements.txt
```

---

### 2. Launch Interactive Web App

Run the real-time Solar Flare Predictor Streamlit application:

```bash
streamlit run app.py
```

Features:
- Select from sample Major Flare events or Quiet Sun background events.
- Upload custom 1-hour sequence inputs (`.npy` or `.csv` of shape `3600 x 4`).
- Select model architectures and adjust operational warning decision thresholds.
- Interactive multi-channel X-ray flux time-series visualization.

---

### 3. Train & Evaluate Deep Learning Models

To train all candidate deep learning architectures using Stratified 5-Fold Cross-Validation:

```bash
python scripts/train_architecture_comparison.py
```

To run individual training experiments:
```bash
python scripts/train_cnn_5fold.py
python scripts/train_cnn_lstm_expanded.py
```

---

### 4. Run Model Explainability & Feature Ablation

To compute Integrated Gradients and channel/temporal attribution:

```bash
python scripts/run_explainability.py
```

To execute feature ablation experiments across detector channels:

```bash
python scripts/run_feature_ablation.py
```

---

## 🧪 Verification & Testing

To verify PyTorch model loading and Streamlit app setup:

```bash
python scripts/test_streamlit_app.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
