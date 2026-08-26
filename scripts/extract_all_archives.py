"""Automated Extraction Pipeline for All Raw SoLEXS & HEL1OS Archives.

Extracts all `.lc.zip` files from `data/solexs/raw/` into `data/solexs/extracted/`
and all Level-1 HEL1OS `.zip` archives from `data/hel1os/raw/` into `data/hel1os/extracted/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from utils import HEL1OS_EXTRACTED_DIR, HEL1OS_RAW_DIR, PROJECT_ROOT, SOLEXS_EXTRACTED_DIR, SOLEXS_RAW_DIR


def extract_solexs_raw() -> int:
    """Extract all SoLEXS ZIP archives into extracted directory."""
    SOLEXS_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = list(SOLEXS_RAW_DIR.rglob("*.zip"))
    print(f"[INFO] Found {len(zip_files)} raw SoLEXS ZIP archives in {SOLEXS_RAW_DIR}...")

    extracted_count = 0
    for z in zip_files:
        try:
            date_dir = SOLEXS_EXTRACTED_DIR / z.parent.name
            date_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(date_dir)
                extracted_count += len(zip_ref.namelist())
        except Exception as err:
            print(f"[WARNING] Error unzipping {z.name}: {err}")

    print(f"[SUCCESS] Extracted {extracted_count} SoLEXS light curve files across {len(zip_files)} archives.")
    return extracted_count


def extract_hel1os_raw() -> int:
    """Extract all HEL1OS Level-1 ZIP archives into extracted directory."""
    HEL1OS_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = list(HEL1OS_RAW_DIR.rglob("*.zip"))
    print(f"[INFO] Found {len(zip_files)} raw HEL1OS ZIP archives in {HEL1OS_RAW_DIR}...")

    extracted_count = 0
    for z in zip_files:
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith(".fits"):
                        out_p = HEL1OS_EXTRACTED_DIR / Path(member).name
                        if not out_p.exists():
                            with zip_ref.open(member) as src, open(out_p, "wb") as dst:
                                dst.write(src.read())
                            extracted_count += 1
        except Exception as err:
            print(f"[WARNING] Error unzipping {z.name}: {err}")

    print(f"[SUCCESS] Extracted {extracted_count} HEL1OS FITS files into {HEL1OS_EXTRACTED_DIR}.")
    return extracted_count


if __name__ == "__main__":
    extract_solexs_raw()
    extract_hel1os_raw()
