"""Safely extract compressed SoLEXS files from raw/ into extracted/."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from utils import SOLEXS_EXTRACTED_DIR, SOLEXS_RAW_DIR


def extract_file(source: Path, destination: Path, overwrite: bool = False) -> bool:
    """Extract one gzip file; return True only when a file was written."""
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as compressed, destination.open("wb") as output:
        shutil.copyfileobj(compressed, output)
    return True


def extract_all(overwrite: bool = False, pattern: str = "*.lc.gz") -> tuple[int, int]:
    """Extract matching `.gz` files (defaulting to light curves `.lc.gz`), retaining date folders."""
    written = skipped = 0
    for source in sorted(SOLEXS_RAW_DIR.rglob(pattern)):
        # Skip heavy spectral PI files by default to preserve disk space
        if pattern == "*.gz" and source.name.endswith(".pi.gz"):
            continue
        relative = source.relative_to(SOLEXS_RAW_DIR)
        destination = SOLEXS_EXTRACTED_DIR / relative.with_suffix("")
        if extract_file(source, destination, overwrite=overwrite):
            written += 1
        else:
            skipped += 1
    return written, skipped



if __name__ == "__main__":
    extracted, skipped = extract_all()
    print(f"Extracted {extracted} file(s); skipped {skipped} existing file(s).")
