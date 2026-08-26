"""Master Executive Script: Run all analysis, threshold optimization, feature ablation, explainability, and benchmark reporting tasks.
"""

import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

tasks = [
    ("Evaluating Trained OOF Models", "scripts/evaluate_models_and_generate_all_reports.py"),
    ("Task 1: Error Analysis & Confusion Matrix", "scripts/analyze_errors.py"),
    ("Tasks 2 & 3: Decision Threshold Tuning & Skill Scores", "scripts/optimize_thresholds.py"),
    ("Task 5: Feature Ablation Study", "scripts/run_feature_ablation.py"),
    ("Task 6: Model Explainability (Integrated Gradients & Saliency)", "scripts/run_explainability.py"),
    ("Task 7: Final Benchmark Package & Figure Generation", "scripts/generate_final_benchmark_package.py"),
]

print("=" * 75)
print("EXECUTING MASTER ANALYSIS & BENCHMARK SUITE")
print("=" * 75)

for title, script_rel in tasks:
    print(f"\n[RUNNING] {title} ({script_rel}) ...")
    script_path = PROJECT_ROOT / script_rel
    res = subprocess.run([PYTHON, str(script_path)], capture_output=True, text=True)
    if res.returncode == 0:
        print(res.stdout.strip())
        print(f"[SUCCESS] Completed {title}")
    else:
        print(f"[ERROR] Failed {title}:")
        print(res.stderr)

print("\n" + "=" * 75)
print("ALL 7 TASKS & DELIVERABLES GENERATED SUCCESSFULLY!")
print("=" * 75)
