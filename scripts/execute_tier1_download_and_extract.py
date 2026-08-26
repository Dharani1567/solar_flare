"""Production Pipeline to Download, Extract, Clean up Tier-1 HEL1OS Dates, and Rebuild Sequence Dataset."""

from __future__ import annotations

import ast
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DOWNLOADS_DIR = PROJECT_ROOT / "data" / "hel1os" / "raw_downloads"
HEL1OS_RAW_DIR = PROJECT_ROOT / "data" / "hel1os" / "raw"
HEL1OS_EXTRACTED_DIR = PROJECT_ROOT / "data" / "hel1os" / "extracted"
ML_DIR = PROJECT_ROOT / "data" / "ml"
RESULTS_DIR = PROJECT_ROOT / "results"

TIER1_TARGET_DATES = [
    "20240514", "20240523", "20240519", "20240524",
    "20240522", "20240515", "20240517", "20240521", "20240516"
]

MAX_WORKERS = 10


def get_active_cookie() -> str:
    """Fetch active authentication cookies from latest script."""
    cookie_script = RAW_DOWNLOADS_DIR / "hel1os_2026Aug26T084448022.py"
    code = cookie_script.read_text(encoding="utf-8")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "cookie_string" and isinstance(node.value, ast.Constant):
                    return str(node.value.value)
    return ""


def get_tier1_urls() -> List[Tuple[str, str]]:
    """Parse relative URLs for Tier-1 dates from hel1os_2026Aug25T033548292.py."""
    source_script = RAW_DOWNLOADS_DIR / "hel1os_2026Aug25T033548292.py"
    code = source_script.read_text(encoding="utf-8")
    tree = ast.parse(code)
    paths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "data_file_paths" and isinstance(node.value, ast.List):
                    paths = [str(elt.value) for elt in node.value.elts if isinstance(elt, ast.Constant)]

    tier1_urls = []
    for p in paths:
        m = re.search(r"level1/(\d{4})/(\d{2})/(\d{2})/", p) or re.search(r"HLS_(\d{4})(\d{2})(\d{2})_", p)
        if m:
            dt = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            if dt in TIER1_TARGET_DATES:
                tier1_urls.append((dt, p))

    return tier1_urls


def process_single_file(
    idx: int,
    total: int,
    dt: str,
    rel_path: str,
    url_prefix: str,
    headers: dict,
    print_lock: threading.Lock,
) -> Tuple[str, str, bool, bool, int, int, str]:
    """Download, verify, extract, and clean up a single ZIP archive."""
    url = url_prefix + rel_path
    filename = rel_path.split("/")[-1].split("?")[0]

    raw_date_dir = HEL1OS_RAW_DIR / dt
    raw_date_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = raw_date_dir / filename

    extracted_date_dir = HEL1OS_EXTRACTED_DIR / dt
    extracted_date_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    bytes_dl = 0
    err_msg = ""
    dl_success = False

    dl_start = time.time()
    try:
        r = session.get(url, headers=headers, stream=True, timeout=(30, 300))
        if r.status_code == 200:
            with open(zip_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bytes_dl += len(chunk)
            dl_success = True
        else:
            err_msg = f"HTTP {r.status_code}"
    except Exception as err:
        err_msg = str(err)

    dl_elapsed = max(time.time() - dl_start, 0.001)

    if not dl_success:
        with print_lock:
            print(f"[{idx}/{total}] [ERROR] Download failed for {dt}/{filename}: {err_msg}")
        return dt, filename, False, False, bytes_dl, 0, err_msg

    # Verify Archive
    is_valid = False
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            is_valid = zf.testzip() is None
    except Exception as e:
        is_valid = False
        err_msg = f"Corrupt ZIP: {e}"

    if not is_valid:
        with print_lock:
            print(f"[{idx}/{total}] [ERROR] ZIP verification failed for {dt}/{filename}: {err_msg}")
        if zip_dest.exists():
            zip_dest.unlink()
        return dt, filename, True, False, bytes_dl, 0, err_msg

    # Extract FITS
    extracted_fits_count = 0
    extract_success = False
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                extracted_path = zf.extract(member, path=extracted_date_dir)
                if extracted_path.endswith(".fits"):
                    extracted_fits_count += 1
        extract_success = True

        # Immediate Cleanup of ZIP file
        if zip_dest.exists():
            zip_dest.unlink()

        with print_lock:
            print(
                f"[{idx}/{total}] [OK] {dt}/{filename} ({bytes_dl/(1024*1024):.1f} MB) -> Extracted {extracted_fits_count} FITS -> ZIP deleted"
            )

    except Exception as err:
        err_msg = f"Extraction error: {err}"
        with print_lock:
            print(f"[{idx}/{total}] [ERROR] Extraction failed for {dt}/{filename}: {err_msg}")

    return dt, filename, dl_success, extract_success, bytes_dl, extracted_fits_count, err_msg


def main():
    print("=" * 70)
    print("TIER-1 HEL1OS PRADAN BULK DOWNLOAD, EXTRACTION & DATASET REBUILD PIPELINE")
    print("=" * 70)

    cookie = get_active_cookie()
    if not cookie:
        print("[ERROR] Could not extract active cookie.")
        sys.exit(1)

    tier1_urls = get_tier1_urls()
    print(f"Total Tier-1 URLs to download: {len(tier1_urls)} across {len(TIER1_TARGET_DATES)} observation dates")

    headers = {"Cookie": cookie, "User-Agent": "Mozilla/5.0"}
    url_prefix = "https://pradan1.issdc.gov.in"
    print_lock = threading.Lock()

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_single_file,
                i + 1,
                len(tier1_urls),
                dt,
                rel_path,
                url_prefix,
                headers,
                print_lock,
            )
            for i, (dt, rel_path) in enumerate(tier1_urls)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    total_time = time.time() - start_time
    total_bytes = sum(r[4] for r in results)
    total_fits = sum(r[5] for r in results if r[3])

    print("\n" + "=" * 70)
    print("DOWNLOAD & EXTRACTION PIPELINE COMPLETED")
    print(f"Total Data Downloaded : {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.3f} GB)")
    print(f"Total FITS Extracted   : {total_fits} files")
    print(f"Pipeline Duration     : {total_time:.1f} seconds ({total_time / 60:.2f} minutes)")
    print("=" * 70)

    # Rebuild Sequence Dataset
    print("\n[STEP 2] Re-building ML Sequence Tensors (build_sequence_dataset_expanded.py)...")
    from build_sequence_dataset_expanded import build_expanded_sequence_dataset

    X, y, meta_df = build_expanded_sequence_dataset()

    print("\n" + "=" * 70)
    print("FINAL REBUILT DATASET SUMMARY")
    print(f"X Tensor Shape        : {X.shape}")
    print(f"y Tensor Shape        : {y.shape}")
    print(f"Total Sequences ($N$) : {len(y)}")
    print(f"Major Flares (M/X)    : {int(np.sum(y == 1))}")
    print(f"Minor Flares (C/B/U)  : {int(np.sum(y == 0))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
