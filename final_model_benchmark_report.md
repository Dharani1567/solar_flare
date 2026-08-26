# Final Model Benchmark Report (Completed Models)

**Dataset**: Frozen Raw 1 Hz HEL1OS Time-Series Tensor (`X_sequences_expanded.npy`)  
**Sequence Tensor Count ($N$)**: `732` (Shape: `(732, 3600, 4)`)  
**Class Breakdown**: `618` Minor Flares ($y=0$), `114` Major Flares ($y=1$, M/X class)  
**Evaluation Protocol**: Stratified 5-Fold Cross-Validation  
**Report Location**: [final_model_benchmark_report.md](file:///home/dharani/Desktop/solar_flare/results/final_model_benchmark_report.md)  

---

## 1. Consolidated Benchmark Comparison Table

| Model Architecture | Parameters | Train Time | Accuracy | Precision | Recall (POD) | F1-Score | ROC-AUC | PR-AUC | TSS | HSS | FAR | CSI | Strengths | Weaknesses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **1D CNN Baseline** | `47,425` | `224.5s` | **0.6803** | **0.3137** | **0.8860** | **0.4633** | **0.8798** | **0.6430** | **0.5284** | **0.3030** | **0.6863** | **0.3015** | Fast inference & feature extraction baseline | Lacks long-range temporal sequence memory over 3600s |
| **CNN + BiLSTM Hybrid** | `198,721` | `1,020.4s` | **0.8388** | **0.4897** | **0.8333** | **0.6169** | **0.9054** | **0.6309** | **0.6731** | **0.5234** | **0.5103** | **0.4460** | Superior TSS (0.6731) and high recall (83.33%) for M/X flares | Slower training speed due to sequential recurrent steps |
| **CNN + Attention + BiLSTM** | `207,041` | `630.3s` | **0.8607** | **0.5448** | **0.6404** | **0.5887** | **0.8870** | **0.6532** | **0.5416** | **0.5055** | **0.4552** | **0.4171** | Highest accuracy (86.07%) & precision (54.48%) via attention weights | Requires threshold tuning (0.35) for maximum recall |

---

## 2. Key Findings & Architectural Trade-Offs

1. **Top Skill Model (CNN + BiLSTM Hybrid)**:
   - Achieved the highest True Skill Statistic (**TSS = 0.6731**) and high recall (**POD = 83.33%**), detecting 95 out of 114 M/X major flares.
   - Preserves sequential memory over the 3,600-second lookback window.

2. **Top Accuracy Model (CNN + Attention + BiLSTM)**:
   - Achieved the highest overall classification accuracy (**86.07%**) and precision (**54.48%**).
   - Multi-head temporal self-attention weights dynamic pre-flare acceleration segments.

3. **Baseline Model (1D CNN)**:
   - Fastest training time (**224.5s**), serving as a fast baseline feature extractor.

---

## 3. Publication Figures

- **Confusion Matrices**: [confusion_matrices_3models.png](file:///home/dharani/Desktop/solar_flare/results/figures/confusion_matrices_3models.png)
- **ROC Curves**: [roc_curves_3models.png](file:///home/dharani/Desktop/solar_flare/results/figures/roc_curves_3models.png)
- **Precision-Recall Curves**: [pr_curves_3models.png](file:///home/dharani/Desktop/solar_flare/results/figures/pr_curves_3models.png)
- **Model Comparison Bar Chart**: [model_comparison_bar_chart.png](file:///home/dharani/Desktop/solar_flare/results/figures/model_comparison_bar_chart.png)
