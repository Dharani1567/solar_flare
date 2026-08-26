"""Copy report deliverables to root directory and assemble a master combined research package document.
"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

reports = [
    "final_model_benchmark_report.md",
    "literature_review_material.md",
    "literature_review_ppt_outline.md",
    "paper_outline.md",
    "confusion_matrix_report.md",
    "error_analysis_report.md",
    "threshold_optimization_report.md",
    "solar_forecast_metrics_report.md",
]

print("Copying individual report files to project root...")
for rep in reports:
    src = RESULTS_DIR / rep
    dst = PROJECT_ROOT / rep
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  -> Copied {src.name} to project root.")

# Create Master Combined Research Package Document
master_file = PROJECT_ROOT / "solar_flare_forecasting_complete_research_package.md"
master_results_file = RESULTS_DIR / "solar_flare_forecasting_complete_research_package.md"

content = """# ADITYA-L1 SOLAR FLARE FORECASTING COMPLETE RESEARCH PACKAGE

**Project**: Solar Flare Forecasting using Aditya-L1 SoLEXS & HEL1OS 1 Hz X-Ray Spectrometer Data  
**Dataset**: Frozen Time-Series Tensor (`X_sequences_expanded.npy`, shape: `(732, 3600, 4)`)  
**Mission**: India's First Solar Observatory at Sun-Earth Lagrangian Point L1  

---

"""

for rep in [
    "final_model_benchmark_report.md",
    "literature_review_material.md",
    "literature_review_ppt_outline.md",
    "paper_outline.md",
    "error_analysis_report.md",
    "threshold_optimization_report.md",
]:
    src = RESULTS_DIR / rep
    if src.exists():
        content += f"\n\n{'='*80}\n"
        content += src.read_text(encoding="utf-8")
        content += f"\n\n{'='*80}\n"

master_file.write_text(content, encoding="utf-8")
master_results_file.write_text(content, encoding="utf-8")

print(f"[SUCCESS] Created Master Combined Research Package Document at:")
print(f"  - {master_file}")
print(f"  - {master_results_file}")
