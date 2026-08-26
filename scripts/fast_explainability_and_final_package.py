"""Immediate Deliverables Runner for Task 6 and Task 7.
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

print("Running Task 6: Model Explainability Analysis...")
res6 = subprocess.run([PYTHON, str(PROJECT_ROOT / "scripts" / "run_explainability.py")], capture_output=True, text=True)
print(res6.stdout)
if res6.stderr:
    print("Stderr:", res6.stderr)

print("Running Task 7: Final Publication Benchmark Package & Figures...")
res7 = subprocess.run([PYTHON, str(PROJECT_ROOT / "scripts" / "generate_final_benchmark_package.py")], capture_output=True, text=True)
print(res7.stdout)
if res7.stderr:
    print("Stderr:", res7.stderr)

print("All reports and figures generated!")
