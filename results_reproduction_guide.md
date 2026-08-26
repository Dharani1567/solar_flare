# Results & Publication Figures Reproduction Guide

**Report Location**: `results/results_reproduction_guide.md`  

---

## Commands to Reproduce All Metrics and Publication Figures

```bash
# 1. Train models and generate out-of-fold predictions
python scripts/train_architecture_comparison.py

# 2. Compute error analysis, threshold sweeps, and benchmark tables
python scripts/evaluate_models_and_generate_all_reports.py
python scripts/analyze_errors.py
python scripts/optimize_thresholds.py
python scripts/run_feature_ablation.py
python scripts/run_explainability.py

# 3. Render all publication-quality figures
python scripts/generate_literature_and_benchmark_deliverables.py
python scripts/generate_paper_figures_extended.py
```
