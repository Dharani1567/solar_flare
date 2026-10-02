#!/usr/bin/env python3
"""
Kaggle Export Package Creator for Dataset v5 & 7-Model Training Pipeline.
Creates solar_flare_dataset_v5.zip and solar_flare_kaggle_package.zip for upload to Kaggle GPU cloud.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = DATA_DIR / "ml"
META_DIR = DATA_DIR / "metadata"

DATASET_ZIP_PATH = PROJECT_ROOT / "solar_flare_dataset_v5.zip"
PACKAGE_ZIP_PATH = PROJECT_ROOT / "solar_flare_kaggle_package.zip"

def create_kaggle_packages():
    print("=" * 75)
    print(" CREATING KAGGLE UPLOAD PACKAGES (V5 DATASET & PIPELINE)")
    print("=" * 75)

    v5_files = [
        (ML_DIR / "X_sequences_v5.npy", "X_sequences_v5.npy"),
        (ML_DIR / "y_labels_v5.npy", "y_labels_v5.npy"),
        (ML_DIR / "sequence_metadata_v5.csv", "sequence_metadata_v5.csv"),
        (META_DIR / "dataset_statistics_v5.json", "dataset_statistics_v5.json"),
        (DATA_DIR / "KAGGLE_README.md", "README.md"),
    ]

    # 1. Create solar_flare_dataset_v5.zip
    print(f"[PACKAGING] Creating {DATASET_ZIP_PATH.name}...")
    with zipfile.ZipFile(DATASET_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in v5_files:
            if src.exists():
                zf.write(src, arcname=arcname)
                print(f"  + Added {arcname} ({src.stat().st_size / (1024**2):.2f} MB)")
            else:
                print(f"  - Warning: Missing {src}")

    sz_ds = DATASET_ZIP_PATH.stat().st_size / (1024**2)
    print(f"[SUCCESS] Saved {DATASET_ZIP_PATH.name} ({sz_ds:.2f} MB)")

    # 2. Create solar_flare_kaggle_package.zip (Full standalone training bundle)
    print(f"\n[PACKAGING] Creating {PACKAGE_ZIP_PATH.name}...")
    package_files = [
        (ML_DIR / "X_sequences_v5.npy", "dataset_v5/X_sequences_v5.npy"),
        (ML_DIR / "y_labels_v5.npy", "dataset_v5/y_labels_v5.npy"),
        (ML_DIR / "sequence_metadata_v5.csv", "dataset_v5/sequence_metadata_v5.csv"),
        (META_DIR / "dataset_statistics_v5.json", "dataset_v5/dataset_statistics_v5.json"),
        (PROJECT_ROOT / "scripts" / "models.py", "models.py"),
        (PROJECT_ROOT / "scripts" / "train_all_models.py", "train_all_models.py"),
        (PROJECT_ROOT / "requirements.txt", "requirements.txt"),
        (PROJECT_ROOT / "README.md", "README.md"),
    ]

    with zipfile.ZipFile(PACKAGE_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in package_files:
            if src.exists():
                zf.write(src, arcname=arcname)
                print(f"  + Added {arcname} ({src.stat().st_size / (1024**2):.2f} MB)")
            else:
                print(f"  - Note: {src.name} will be created before final zip.")

    sz_pkg = PACKAGE_ZIP_PATH.stat().st_size / (1024**2)
    print(f"[SUCCESS] Saved {PACKAGE_ZIP_PATH.name} ({sz_pkg:.2f} MB)")

if __name__ == "__main__":
    create_kaggle_packages()
