"""HEL1OS FITS Data Loader & Parsing Helper.

Loads HEL1OS light curve and spectral observation products from extracted FITS files
or raw ZIP archives into pandas DataFrames with UTC timestamps and standard column names.
"""

from __future__ import annotations

import io
from pathlib import Path
import zipfile
from astropy.io import fits
import numpy as np
import pandas as pd

from utils import HEL1OS_EXTRACTED_DIR, HEL1OS_RAW_DIR, PROJECT_ROOT


def list_hel1os_files(extracted_dir: Path | None = None) -> list[Path]:
    """Find all extracted HEL1OS light curve FITS files."""
    base_dir = extracted_dir or HEL1OS_EXTRACTED_DIR
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("lightcurve_*.fits"))


def load_hel1os_light_curve(
    fits_source: Path | str | bytes,
    hdu_name: str | None = None,
    energy_band: str = "wide",
) -> pd.DataFrame:
    """Load a HEL1OS light curve FITS product into a clean pandas DataFrame.

    Parameters
    ----------
    fits_source : Path | str | bytes
        Path to extracted FITS file, path to ZIP file, or FITS byte buffer.
    hdu_name : str | None
        Specific HDU extension name. If None, auto-selects based on energy_band.
    energy_band : str
        Energy band shortcut: 'wide' (default), '5-20', '20-30', '30-40', '40-60', '20-40', '60-80', '80-150'.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        ['time', 'time_dt', 'counts', 'error', 'mjd', 'isot', 'detector', 'energy_band']
    """
    close_hdul = False

    if isinstance(fits_source, (str, Path)):
        p = Path(fits_source)
        if p.suffix == ".zip":
            # Search for light curve inside ZIP
            with zipfile.ZipFile(p, "r") as zf:
                lc_members = [m for m in zf.namelist() if "lightcurve_" in m and m.endswith(".fits")]
                if not lc_members:
                    raise FileNotFoundError(f"No light curve FITS found inside {p.name}")
                fits_bytes = zf.read(lc_members[0])
                hdul = fits.open(io.BytesIO(fits_bytes))
                close_hdul = True
        else:
            hdul = fits.open(p)
            close_hdul = True
    elif isinstance(fits_source, bytes):
        hdul = fits.open(io.BytesIO(fits_source))
        close_hdul = True
    else:
        hdul = fits_source

    try:
        hdu_list = [h.name for h in hdul if hasattr(h, "columns")]

        # Determine target HDU extension name
        selected_hdu_name = None

        if hdu_name and hdu_name in hdu_list:
            selected_hdu_name = hdu_name
        else:
            # Auto-selection strategy
            if energy_band.lower() == "wide":
                wide_candidates = [h for h in hdu_list if "1.80KEV_TO_90" in h or "18.00KEV_TO_160" in h]
                selected_hdu_name = wide_candidates[0] if wide_candidates else hdu_list[0]
            else:
                band_tag = energy_band.replace("-", ".00KEV_TO_").upper()
                matching = [h for h in hdu_list if band_tag in h or energy_band in h]
                selected_hdu_name = matching[0] if matching else hdu_list[0]

        target_hdu = hdul[selected_hdu_name]
        raw_table = pd.DataFrame(target_hdu.data)

        # Standardize columns
        df = pd.DataFrame()
        if "ISOT" in raw_table.columns:
            df["isot"] = raw_table["ISOT"].astype(str).str.strip()
            df["time_dt"] = pd.to_datetime(df["isot"], utc=True)
            df["time"] = (df["time_dt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        elif "MJD" in raw_table.columns:
            df["mjd"] = raw_table["MJD"].astype(float)
            # Convert MJD to Unix timestamp: (MJD - 40587) * 86400
            df["time"] = (df["mjd"] - 40587.0) * 86400.0
            df["time_dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["isot"] = df["time_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

        if "MJD" in raw_table.columns:
            df["mjd"] = raw_table["MJD"].astype(float)

        df["counts"] = raw_table["CTR"].astype(float) if "CTR" in raw_table.columns else 0.0
        df["error"] = raw_table["STAT_ERR"].astype(float) if "STAT_ERR" in raw_table.columns else 0.0

        # Infer detector and energy band metadata from HDU name
        df["hdu_name"] = selected_hdu_name
        df["detector"] = "CDTE" if "CDTE" in selected_hdu_name else ("CZT" if "CZT" in selected_hdu_name else "HEL1OS")
        df["energy_band"] = selected_hdu_name

        return df

    finally:
        if close_hdul and hdul:
            hdul.close()


def get_hel1os_observation_summary(fits_path: Path) -> dict:
    """Compute observation statistics including timestamps, sampling cadence, and record count."""
    df = load_hel1os_light_curve(fits_path)

    time_diffs = df["time"].diff().dropna()
    mean_cadence = float(time_diffs.mean()) if len(time_diffs) > 0 else 0.0
    std_cadence = float(time_diffs.std()) if len(time_diffs) > 0 else 0.0

    return {
        "file_name": fits_path.name,
        "detector": df["detector"].iloc[0] if not df.empty else "UNKNOWN",
        "record_count": len(df),
        "t_start_utc": str(df["time_dt"].min()) if not df.empty else "",
        "t_end_utc": str(df["time_dt"].max()) if not df.empty else "",
        "duration_sec": float(df["time"].max() - df["time"].min()) if not df.empty else 0.0,
        "cadence_sec_mean": round(mean_cadence, 3),
        "cadence_sec_std": round(std_cadence, 3),
        "min_counts": float(df["counts"].min()) if not df.empty else 0.0,
        "max_counts": float(df["counts"].max()) if not df.empty else 0.0,
        "mean_counts": float(df["counts"].mean()) if not df.empty else 0.0,
    }


if __name__ == "__main__":
    files = list_hel1os_files()
    print(f"Found {len(files)} extracted HEL1OS light curve files.")

    if files:
        sample_file = files[0]
        print(f"\n--- Loading sample observation: {sample_file.name} ---")
        df = load_hel1os_light_curve(sample_file)
        print(df.head(10))

        summary = get_hel1os_observation_summary(sample_file)
        print("\n--- Observation Summary ---")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")
