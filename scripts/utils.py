"""Shared path and file-discovery helpers for the solar flare project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SOLEXS_DIR = PROJECT_ROOT / "data" / "solexs"
SOLEXS_RAW_DIR = SOLEXS_DIR / "raw"
SOLEXS_EXTRACTED_DIR = SOLEXS_DIR / "extracted"
CATALOG_DIR = SOLEXS_DIR / "flare_catalog"
MERGED_CATALOG_FILE = CATALOG_DIR / "flare_events_merged.csv"
LABELED_CATALOG_FILE = CATALOG_DIR / "flare_events_labeled.csv"
GOES_CATALOG_FILE = CATALOG_DIR / "goes_catalog_raw.csv"
GOES_PLOT_OUTPUT = PROJECT_ROOT / "results" / "goes_class_distribution.png"
HEL1OS_DIR = PROJECT_ROOT / "data" / "hel1os"
HEL1OS_RAW_DIR = HEL1OS_DIR / "raw"
HEL1OS_EXTRACTED_DIR = HEL1OS_DIR / "extracted"
PROCESSED_DIR = HEL1OS_DIR / "processed"


def ensure_output_directories() -> None:
    """Create derived-data directories without touching raw observations."""
    SOLEXS_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)


def solexs_light_curves() -> list[Path]:
    """Return unique SoLEXS light-curve files (deduplicated by filename)."""
    all_files = sorted({*SOLEXS_EXTRACTED_DIR.rglob("*.lc"), *SOLEXS_EXTRACTED_DIR.rglob("*.lc.gz")})
    seen_names: set[str] = set()
    unique_files: list[Path] = []
    for p in all_files:
        if p.name not in seen_names:
            seen_names.add(p.name)
            unique_files.append(p)
    return unique_files



def observation_date(path: Path) -> str:
    """Return the YYYYMMDD date for a file from its parent path or filename."""
    import re
    match = re.search(r"(20\d{6})", str(path))
    if match:
        return match.group(1)
    return path.parent.name

