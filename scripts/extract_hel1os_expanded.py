"""Extract all downloaded HEL1OS Level-1 ZIP archives into data/hel1os/extracted/YYYYMMDD/."""

from __future__ import annotations

from pathlib import Path
import zipfile

from utils import HEL1OS_EXTRACTED_DIR, HEL1OS_RAW_DIR, PROJECT_ROOT


def extract_all_hel1os() -> None:
    zip_files = list(HEL1OS_RAW_DIR.rglob("*.zip"))
    print(f"[INFO] Found {len(zip_files)} raw HEL1OS ZIP archives in {HEL1OS_RAW_DIR}...")

    extracted_count = 0
    for z_path in zip_files:
        # Date string from path or filename
        m = z_path.parent.name
        if not m.isdigit():
            m_date = re.search(r"20\d{6}", z_path.name)
            date_str = m_date.group(0) if m_date else "unknown"
        else:
            date_str = m

        out_dir = HEL1OS_EXTRACTED_DIR / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(z_path, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    out_file = out_dir / member.filename
                    if not out_file.exists() or out_file.stat().st_size == 0:
                        zf.extract(member, path=out_dir)
                        extracted_count += 1
        except Exception as err:
            print(f"[WARNING] Extraction error for {z_path.name}: {err}")

    print(f"[SUCCESS] Extracted {extracted_count} FITS files into {HEL1OS_EXTRACTED_DIR}")


if __name__ == "__main__":
    import re
    extract_all_hel1os()
