"""HEL1OS ZIP Extraction Pipeline Module.

Extracts downloaded HEL1OS Level-1 ZIP archives from `data/hel1os/raw/YYYYMMDD/`
into `data/hel1os/extracted/YYYYMMDD/`, preserving the directory hierarchy and
FITS product layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from utils import (
    HEL1OS_DIR,
    HEL1OS_EXTRACTED_DIR,
    HEL1OS_RAW_DIR,
    PROJECT_ROOT,
    ensure_output_directories,
)


def extract_hel1os_archive(
    zip_path: Path,
    target_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Extract a single HEL1OS level-1 ZIP archive into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: list[Path] = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                out_path = target_dir / member.filename
                if out_path.exists() and not overwrite and out_path.stat().st_size > 0:
                    extracted_files.append(out_path)
                    continue

                extracted_path = zf.extract(member, path=target_dir)
                extracted_files.append(Path(extracted_path))
    except Exception as err:
        print(f"[ERROR] Failed to extract {zip_path.name}: {err}")

    return extracted_files


def extract_all_hel1os(
    raw_dir: Path | None = None,
    extracted_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Extract all downloaded HEL1OS ZIP archives from raw_dir into extracted_dir."""
    ensure_output_directories()
    raw_dir = raw_dir or HEL1OS_RAW_DIR
    extracted_dir = extracted_dir or HEL1OS_EXTRACTED_DIR

    zip_files = sorted(raw_dir.rglob("*.zip"))
    print(f"[INFO] Found {len(zip_files)} HEL1OS ZIP archives in {raw_dir.relative_to(PROJECT_ROOT)}")

    total_zips = len(zip_files)
    extracted_count = 0
    extracted_fits_count = 0

    for idx, z_path in enumerate(zip_files, 1):
        # Extract date from parent folder or path
        date_folder = z_path.parent.parent.name if z_path.parent.name.startswith("N00_") else z_path.parent.name
        if not date_folder.isdigit() or len(date_folder) != 8:
            # Fallback search for YYYYMMDD
            import re
            m = re.search(r"(20\d{6})", str(z_path))
            date_folder = m.group(1) if m else "unknown_date"

        dest_dir = extracted_dir / date_folder
        print(f"  [{idx}/{total_zips}] Extracting {z_path.name} -> {dest_dir.relative_to(PROJECT_ROOT)}...")
        extracted_files = extract_hel1os_archive(z_path, dest_dir, overwrite=overwrite)
        extracted_count += 1
        fits_in_zip = sum(1 for p in extracted_files if p.name.endswith(".fits"))
        extracted_fits_count += fits_in_zip

    stats = {
        "total_zips": total_zips,
        "extracted_zips": extracted_count,
        "extracted_fits_files": extracted_fits_count,
    }

    print("\n" + "=" * 60)
    print("HEL1OS EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total ZIP Archives Processed : {stats['extracted_zips']}/{stats['total_zips']}")
    print(f"Total Extracted FITS Files  : {stats['extracted_fits_files']}")
    print(f"Output Target Directory     : {extracted_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 60 + "\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEL1OS ZIP Extraction Pipeline")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing extracted files")
    args = parser.parse_args()

    extract_all_hel1os(overwrite=args.overwrite)
