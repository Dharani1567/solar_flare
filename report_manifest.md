# Complete Generated Reports & Manifest Audit

**Report Location**: `results/report_manifest.md`  

---

## Audit of All Generated Project Documentation

| Report Filename | Location | Purpose & Subject Matter | Primary Output Metrics |
| :--- | :--- | :--- | :--- |
| **`final_model_benchmark_report.md`** | `results/` & Root | Multi-architecture performance comparison | Accuracy, Precision, Recall, F1, ROC-AUC, TSS, HSS |
| **`literature_review_material.md`** | `results/` & Root | Literature review & mission background | Aditya-L1 SoLEXS & HEL1OS specs, physical contributions |
| **`literature_review_ppt_outline.md`** | `results/` & Root | Presentation deck structure | 11-slide PPT outline |
| **`paper_outline.md`** | `results/` & Root | Journal paper manuscript outline | 9-section paper structure |
| **`restore_project_guide.md`** | `results/` & Root | Project restoration manual | Step-by-step restoration workflow |
| **`environment_report.md`** | `results/` & Root | Hardware & Python environment audit | Operating system, PyTorch version, `pip freeze` |
| **`data_lineage_report.md`** | `results/` & Root | End-to-end data transformation pipeline | Raw FITS -> Extracted -> Labeled -> Tensors |
| **`dataset_rebuild_guide.md`** | `results/` & Root | Dataset reconstruction workflow | Shell commands to rebuild `X_sequences_expanded.npy` |
| **`model_reconstruction_guide.md`** | `results/` & Root | Model architecture hyperparameters | Neural net layers, loss function, learning rate, CV folds |
| **`threshold_optimization_report.md`** | `results/` & Root | Decision threshold tuning | Threshold sweep ($0.05 ightarrow 0.95$), optimal $0.35$ point |
| **`error_analysis_report.md`** | `results/` & Root | Model error mode investigation | Itemized False Negative & False Positive analysis |
| **`backup_checklist.md`** | `results/` & Root | File preservation priority checklist | Critical vs optional artifacts matrix |
| **`reproducibility_gaps.md`** | `results/` & Root | Reproducibility audit & gap analysis | ISSDC login, satellite gap handling, hardware differences |
