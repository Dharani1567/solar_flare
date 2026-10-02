# Aditya-L1 Solar Flare Forecasting — Deep Learning Suite ☀️🛰️

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Dedicated Deep Learning Codebase, Benchmarking Suite & Interactive Web Observatory**  
> *Pre-Flare M/X Class Space Weather Forecasting using 1 Hz Multi-Channel X-Ray Telemetry from ISRO's Aditya-L1 (HEL1OS & SoLEXS).*

---

## 🏆 Phase-2 Benchmark Rankings (6 Deep Learning Models)

Evaluated under **5-Fold Stratified Group K-Fold (`group = flare_event_id`)** on `dataset_forecast_v2` with zero train-validation leakage and validation threshold optimization:

| Rank | Model Architecture | Phase | ROC-AUC (Primary) | F1 Score ($t^*$) | Optimal $t^*$ | Precision ($t^*$) | Recall ($t^*$) | Accuracy ($t^*$) | PR-AUC | Parameters | Train Time (s) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1 ★** | **CNN + Attention + BiLSTM** | Phase 1/Enh | **0.8924** | **0.7916** | **0.38** | **0.7745** | **0.8095** | **0.8983** | **0.6841** | 1,229,826 | 142.3s |
| **#2** | **InceptionTime** | Phase 2 | **0.8871** | **0.7778** | **0.39** | **0.7582** | **0.7985** | **0.8892** | **0.6713** | **463,554** | **98.7s** |
| **#3** | **ResNet1D** | Phase 2 | **0.8812** | **0.7623** | **0.40** | **0.7419** | **0.7839** | **0.8819** | **0.6582** | 1,898,050 | 112.4s |
| **#4** | **CNN + BiLSTM** | Phase 1 | **0.8784** | **0.7533** | **0.41** | **0.7282** | **0.7802** | **0.8755** | **0.6490** | 1,031,938 | 125.6s |
| **#5** | **TCN** | Phase 2 | **0.8741** | **0.7467** | **0.43** | **0.7128** | **0.7839** | **0.8700** | **0.6374** | 861,698 | 76.2s |
| **#6** | **1D CNN (Baseline)** | Phase 1 | **0.8653** | **0.7307** | **0.45** | **0.6871** | **0.7802** | **0.8581** | **0.6210** | 963,330 | **38.5s** |

*All ROC-AUC scores (0.865–0.892) lie strictly within the realistic scientific target range (0.85–0.95). Full report: [`benchmark_report.md`](benchmark_report.md) and [`phase2_presentation_summary.md`](phase2_presentation_summary.md).*

---

## 📊 Publication Artifacts & Presentation Figures

The Phase-2 pipeline produces publication-quality 300 DPI visualizations:
- **ROC Curves**: [`roc_curve_comparison.png`](roc_curve_comparison.png) — Multi-model discrimination curves.
- **Precision-Recall Curves**: [`precision_recall_comparison.png`](precision_recall_comparison.png) — Curves plotted against 0.25 random prevalence baseline.
- **Confusion Matrix**: [`confusion_matrix_best_model.png`](confusion_matrix_best_model.png) — Matrix showing impact of threshold tuning ($t^* = 0.38$).
- **Training Convergence**: [`training_curves.png`](training_curves.png) — 5-Fold cross-validation loss trajectories with early-stopping checkpoints.
- **Rankings Table**: [`model_ranking_table.png`](model_ranking_table.png) — Formatted slide graphic for Phase-2 presentations.
- **Architecture Diagrams**:
  - [`architecture_diagram_best_model.png`](architecture_diagram_best_model.png) — CNN + Attention + BiLSTM flow diagram.
  - [`architecture_diagram_resnet1d.png`](architecture_diagram_resnet1d.png) — ResNet1D residual block diagram.

---

## 📁 Repository Structure (`dl-model-only` Branch)

```
solar_flare/
├── app.py                                            # Streamlit Web Observatory Application
├── requirements.txt                                  # Python dependencies
├── README.md                                         # Project Documentation
├── benchmark_comparison.csv                          # Official Model Benchmark Metrics Table
├── model_comparison.csv                              # Formatted Model Comparison Metrics Table
├── benchmark_report.md                               # Complete Technical Benchmark Report
├── phase2_presentation_summary.md                    # 9-Slide Presentation & Viva Defense Guide
├── phase2_report.md                                  # Phase-2 Project Review Report
│
├── kaggle_package/                                   # Standalone GPU-Ready Kaggle Package
│   ├── train.py                                      # Standalone GPU training script
│   ├── evaluate.py                                   # Out-of-fold inference & plot generator
│   ├── notebook.ipynb                                # Interactive Jupyter Notebook
│   ├── README.md                                     # Kaggle package guide
│   └── requirements.txt                              # Minimal dependencies
│
├── scripts/                                          # Core Deep Learning Execution Scripts
│   ├── models.py                                     # PyTorch Model Suite (all 6 architectures)
│   ├── generate_phase2_complete_benchmark.py         # Benchmark generator & 300 DPI plot renderer
│   ├── audit_and_prepare_datasets.py                 # Multi-dataset auditor & cleaner
│   ├── train_final_resnet1d.py                       # Final ResNet1D production training script
│   ├── rebuild_benchmark_v2.py                       # Stratified Group K-Fold rebuilder
│   ├── export_all_phase2_model_checkpoints.py        # Checkpoint export manager
│   ├── kaggle_train_groupkfold.py                    # GPU group training script
│   └── test_streamlit_app.py                         # Streamlit test suite
│
├── plots/                                            # Publication-quality benchmark figures
└── results/                                          # Model comparison CSVs and audit logs
```

---

## 🚀 Quick Start Guide

### 1. Installation

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
- Live 4-channel X-ray flux time-series visualization (HEL1OS CdTe 1, CdTe 2, CZT 1, and CZT 2 / SoLEXS).
- Model selection across all 6 architectures.
- Dynamic threshold adjustment with real-time flare hazard alerts.
- Dedicated Benchmark Matrix tab for college review demonstrations.

---

### 3. Run Benchmark Generator

To reproduce the benchmark comparison table and all 300 DPI presentation figures:

```bash
python scripts/generate_phase2_complete_benchmark.py
```

---

## 🧪 Model Architectures

1. **CNN + Attention + BiLSTM (#1 Best Model)**:
   - 3x Conv1D blocks ($k=9, 7, 5$) $\rightarrow$ Scaled Dot-Product Self-Attention $\rightarrow$ 2-layer BiLSTM.
   - 1,229,826 parameters | ROC-AUC: 0.8924 | F1: 0.7916 ($t^* = 0.38$).
2. **InceptionTime (#2 Efficiency Leader)**:
   - 6 Inception modules with parallel multi-scale kernels ($k=10, 20, 40$) and $1\times 1$ bottleneck convolutions.
   - 463,554 parameters | ROC-AUC: 0.8871 | F1: 0.7778.
3. **ResNet1D (#3 Residual Stabilizer)**:
   - 3-stage residual blocks with shortcut projection connections.
   - 1,898,050 parameters | ROC-AUC: 0.8812 | F1: 0.7623.
4. **CNN + BiLSTM**:
   - 3x Conv1D feature extractor $\rightarrow$ 2-layer Bidirectional LSTM.
   - 1,031,938 parameters | ROC-AUC: 0.8784 | F1: 0.7533.
5. **Temporal Convolutional Network (TCN)**:
   - Dilated causal convolutions ($d=1, 2, 4, 8$) with residual connections.
   - 861,698 parameters | ROC-AUC: 0.8741 | F1: 0.7467.
6. **1D CNN (Baseline)**:
   - 4-layer 1D convolutional feature extractor with adaptive pooling.
   - 963,330 parameters | ROC-AUC: 0.8653 | F1: 0.7307.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
