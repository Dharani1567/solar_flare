#!/usr/bin/env python3
"""
Update Kaggle Upload Packages with Leak-Free StratifiedGroupKFold Pipeline & Reports.
"""

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = DATA_DIR / "ml"
META_DIR = DATA_DIR / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

PACKAGE_ZIP_PATH = PROJECT_ROOT / "solar_flare_kaggle_package.zip"

def update_package():
    print("=" * 75)
    print(" REBUILDING KAGGLE PACKAGE WITH LEAK-FREE GROUP-KFOLD PIPELINE")
    print("=" * 75)

    files_to_pack = [
        (ML_DIR / "X_sequences_v5.npy", "dataset_v5/X_sequences_v5.npy"),
        (ML_DIR / "y_labels_v5.npy", "dataset_v5/y_labels_v5.npy"),
        (ML_DIR / "sequence_metadata_v5.csv", "dataset_v5/sequence_metadata_v5.csv"),
        (META_DIR / "dataset_statistics_v5.json", "dataset_v5/dataset_statistics_v5.json"),
        (SCRIPTS_DIR / "models.py", "models.py"),
        (SCRIPTS_DIR / "kaggle_train_groupkfold.py", "kaggle_train_groupkfold.py"),
        (SCRIPTS_DIR / "kaggle_train_groupkfold.py", "train_all_models.py"), # Alias for compatibility
        (RESULTS_DIR / "fold_assignments.csv", "fold_assignments.csv"),
        (PROJECT_ROOT / "requirements.txt", "requirements.txt"),
        (PROJECT_ROOT / "README.md", "README.md"),
    ]

    with zipfile.ZipFile(PACKAGE_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_to_pack:
            if src.exists():
                zf.write(src, arcname=arcname)
                print(f"  + Added {arcname} ({src.stat().st_size / (1024**2):.2f} MB)")
            else:
                print(f"  - Warning: Missing {src.name}")

    sz = PACKAGE_ZIP_PATH.stat().st_size / (1024**2)
    print(f"\n[SUCCESS] Saved updated {PACKAGE_ZIP_PATH.name} ({sz:.2f} MB)")

if __name__ == "__main__":
    update_package()
