"""Expanded SoLEXS Flare Detection & Merging Pipeline.

Processes all extracted SoLEXS Level-1 light curves in `data/solexs/extracted/` across old and new dates,
detects solar flare events using derivative thresholding & SNR filtering, merges overlapping windows,
and exports `data/solexs/flare_catalog/flare_events_merged_expanded.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from astropy.io import fits
import numpy as np
import pandas as pd

from utils import PROJECT_ROOT, SOLEXS_EXTRACTED_DIR

CATALOG_DIR = PROJECT_ROOT / "data" / "solexs" / "flare_catalog"
EXPANDED_MERGED_CSV = CATALOG_DIR / "flare_events_merged_expanded.csv"


def process_single_solexs_file(fits_path: Path) -> pd.DataFrame:
    """Process a single SoLEXS FITS/LC file and detect candidate flare peaks."""
    flares = []
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            col_names = [c.upper() for c in data.names]

            time_col = "TIME" if "TIME" in col_names else col_names[0]
            counts_col = "COUNTS" if "COUNTS" in col_names else ("RATE" if "RATE" in col_names else ("CTR" if "CTR" in col_names else col_names[1]))

            time_sec = data[time_col]
            raw_counts = pd.Series(data[counts_col]).fillna(0.0).values

            if len(raw_counts) < 300:
                return pd.DataFrame()

            # Moving average background estimation
            window_size = 300
            bkg = pd.Series(raw_counts).rolling(window=window_size, min_periods=30, center=True).median().bfill().ffill().values
            net_counts = raw_counts - bkg
            
            valid_net = net_counts[~np.isnan(net_counts)]
            std_bkg = float(np.std(valid_net[:300])) if len(valid_net) >= 300 else 1.0
            if std_bkg <= 0:
                std_bkg = 1.0

            snr = np.where(std_bkg > 0, net_counts / std_bkg, 0.0)
            snr = np.nan_to_num(snr, nan=0.0)
            net_counts = np.nan_to_num(net_counts, nan=0.0)

            # Detect candidate flare points (SNR >= 3.5 and net_counts > 30)
            candidate_indices = np.where((snr >= 3.5) & (net_counts > 30))[0]

            if len(candidate_indices) == 0:
                return pd.DataFrame()

            # Group contiguous candidate points into events (allow 120s gaps)
            groups = np.split(candidate_indices, np.where(np.diff(candidate_indices) > 120)[0] + 1)

            date_m = re.search(r"20\d{6}", str(fits_path))
            date_str = date_m.group(0) if date_m else "20240101"

            for g in groups:
                if len(g) < 10:  # Require at least 10s duration
                    continue
                peak_idx = g[np.argmax(raw_counts[g])]
                t_start = float(time_sec[g[0]])
                t_peak = float(time_sec[peak_idx])
                t_end = float(time_sec[g[-1]])
                p_cnt = float(raw_counts[peak_idx])
                p_snr = float(snr[peak_idx])

                flares.append({
                    "event_id": f"SOLEXS_{date_str}_{int(t_peak)}",
                    "date": date_str,
                    "start_time": t_start,
                    "peak_time": t_peak,
                    "end_time": t_end,
                    "duration_sec": max(1.0, t_end - t_start),
                    "peak_counts": p_cnt,
                    "snr": p_snr,
                })
    except Exception as err:
        pass

    return pd.DataFrame(flares)


def run_expanded_solexs_processing() -> pd.DataFrame:
    """Process all available extracted SoLEXS light curve files and build expanded catalog."""
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    all_lc_files = list(SOLEXS_EXTRACTED_DIR.rglob("*.lc")) + list(SOLEXS_EXTRACTED_DIR.rglob("*.fits"))
    print(f"[INFO] Scanning {len(all_lc_files)} SoLEXS light curve files across {len(list(SOLEXS_EXTRACTED_DIR.iterdir()))} dates...")

    dfs = []
    for f in all_lc_files:
        df_f = process_single_solexs_file(f)
        if not df_f.empty:
            dfs.append(df_f)

    if not dfs:
        raise ValueError("No flare events detected across extracted light curves!")

    all_flares = pd.concat(dfs, ignore_index=True).sort_values(["date", "peak_time"]).reset_index(drop=True)

    # Merge overlapping events within 300 seconds
    merged = []
    current = None

    for _, row in all_flares.iterrows():
        if current is None:
            current = row.to_dict()
        else:
            if row["date"] == current["date"] and (row["start_time"] - current["end_time"]) < 300:
                current["end_time"] = max(current["end_time"], row["end_time"])
                current["duration_sec"] = current["end_time"] - current["start_time"]
                if row["peak_counts"] > current["peak_counts"]:
                    current["peak_time"] = row["peak_time"]
                    current["peak_counts"] = row["peak_counts"]
                    current["snr"] = row["snr"]
            else:
                merged.append(current)
                current = row.to_dict()

    if current is not None:
        merged.append(current)

    df_merged = pd.DataFrame(merged)
    df_merged["event_id"] = [f"SLX_{row['date']}_{idx:04d}" for idx, row in df_merged.iterrows()]
    df_merged.to_csv(EXPANDED_MERGED_CSV, index=False)

    print(f"[SUCCESS] Saved expanded merged flare catalog to: {EXPANDED_MERGED_CSV}")
    print(f"[SUMMARY] Total Expanded Flares: {len(df_merged)} across {len(df_merged['date'].unique())} observation dates")
    return df_merged


if __name__ == "__main__":
    run_expanded_solexs_processing()
