"""Phase 2: Kaggle Dataset Package Generator.

Creates upload-ready zip archive `solar_flare_dataset_v3.zip` containing:
- X_sequences_v3.npy
- y_labels_v3.npy
- sequence_metadata_v3.csv
- dataset_statistics.json
- README.md
- dataset-metadata.json
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = DATA_DIR / "ml"
META_DIR = DATA_DIR / "metadata"
KAGGLE_ZIP_PATH = PROJECT_ROOT / "solar_flare_dataset_v3.zip"


def build_kaggle_package():
    print("=" * 75)
    print("PHASE 2: PREPARING KAGGLE UPLOAD PACKAGE")
    print("=" * 75)

    # 1. Create dataset-metadata.json for Kaggle API
    kaggle_meta = {
        "title": "Aditya-L1 Solar Flare Forecasting Dataset v3",
        "id": "isro-aditya-l1-solar-flare-v3",
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": "1 Hz 4-channel X-ray time-series dataset from ISRO Aditya-L1 HEL1OS & SoLEXS spectrometers for major solar flare forecasting.",
    }
    kaggle_meta_path = DATA_DIR / "dataset-metadata.json"
    kaggle_meta_path.write_text(json.dumps(kaggle_meta, indent=2), encoding="utf-8")

    # 2. Create README.md for Kaggle Dataset
    readme_content = """# ISRO Aditya-L1 Solar Flare Forecasting Dataset (v3)

## Dataset Specifications
- **Tensor Shape**: `(732, 3600, 4)` (Samples x Seconds x Channels)
- **Class Breakdown**: 114 Major M/X Flares (`y=1`), 618 Minor / Quiet Events (`y=0`)
- **Imbalance Ratio**: 5.42 : 1
- **Channels**:
  1. `CdTe 1`: High Energy Spectrometer 1 (10-150 keV)
  2. `CdTe 2`: High Energy Spectrometer 2 (10-150 keV)
  3. `CZT 1`: CZT Spectrometer Channel 1
  4. `CZT 2`: CZT Spectrometer Channel 2

## Usage in Python
```python
import numpy as np
X = np.load('X_sequences_v3.npy')
y = np.load('y_labels_v3.npy')
print("X shape:", X.shape, "y shape:", y.shape)
```
"""
    readme_path = DATA_DIR / "KAGGLE_README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    # 3. Compress into solar_flare_dataset_v3.zip
    files_to_zip = [
        (ML_DIR / "X_sequences_v3.npy", "X_sequences_v3.npy"),
        (ML_DIR / "y_labels_v3.npy", "y_labels_v3.npy"),
        (ML_DIR / "sequence_metadata_v3.csv", "sequence_metadata_v3.csv"),
        (META_DIR / "dataset_statistics.json", "dataset_statistics.json"),
        (kaggle_meta_path, "dataset-metadata.json"),
        (readme_path, "README.md"),
    ]

    with zipfile.ZipFile(KAGGLE_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_path, arc_name in files_to_zip:
            if src_path.exists():
                zf.write(src_path, arcname=arc_name)
                print(f"  [ZIP ADD] {arc_name} ({src_path.stat().st_size / (1024**2):.2f} MB)")

    print(f"\n[SUCCESS] Exported Kaggle Package: {KAGGLE_ZIP_PATH}")
    print(f"  Total Zip Size: {KAGGLE_ZIP_PATH.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    build_kaggle_package()
