"""Analysis of flare events excluded due to missing HEL1OS coverage.

Parses expanded catalog, checks extracted HEL1OS light curve FITS files,
checks PRADAN download script URLs, categorizes missing reason and channels,
generates a comprehensive recoverability report and CSV.
"""

from __future__ import annotations
import ast
import re
from pathlib import Path
import numpy as np
import pandas as pd

from utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    HEL1OS_EXTRACTED_DIR,
    HEL1OS_RAW_DIR,
)
from load_hel1os import list_hel1os_files, load_hel1os_light_curve

EXPANDED_LABELED_CSV = PROJECT_ROOT / "data" / "solexs" / "flare_catalog" / "flare_events_labeled_expanded.csv"
RAW_DOWNLOADS_DIR = PROJECT_ROOT / "data" / "hel1os" / "raw_downloads"
REPORT_MD = RESULTS_DIR / "hel1os_recoverability_report.md"
RECOVERABILITY_CSV = RESULTS_DIR / "hel1os_excluded_events_analysis.csv"


def get_pradan_urls_and_dates() -> tuple[set[str], dict[str, list[str]]]:
    """Parse PRADAN scripts in raw_downloads to get available URLs and dates."""
    pradan_dates = set()
    date_to_urls = {}

    for p in RAW_DOWNLOADS_DIR.glob("*.py"):
        try:
            code = p.read_text(encoding="utf-8")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "data_file_paths":
                            if isinstance(node.value, ast.List):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        url_path = str(elt.value)
                                        # Extract date YYYYMMDD
                                        m = re.search(r"HLS_(\d{8})_", url_path)
                                        if not m:
                                            m = re.search(r"level1/(\d{4})/(\d{2})/(\d{2})/", url_path)
                                            if m:
                                                dt_str = f"{m.group(1)}{m.group(2)}{m.group(3)}"
                                            else:
                                                m2 = re.search(r"(20\d{6})", url_path)
                                                dt_str = m2.group(1) if m2 else ""
                                        else:
                                            dt_str = m.group(1)
                                        
                                        if dt_str:
                                            pradan_dates.add(dt_str)
                                            date_to_urls.setdefault(dt_str, []).append(url_path)
        except Exception as e:
            print(f"[WARN] Error parsing {p.name}: {e}")

    return pradan_dates, date_to_urls


def analyze_excluded_events():
    df_flares = pd.read_csv(EXPANDED_LABELED_CSV)
    print(f"[INFO] Total flare events in expanded catalog: {len(df_flares)}")

    all_lc_files = list_hel1os_files(HEL1OS_EXTRACTED_DIR)
    print(f"[INFO] Total extracted HEL1OS FITS files found: {len(all_lc_files)}")

    pradan_dates, date_to_urls = get_pradan_urls_and_dates()
    print(f"[INFO] PRADAN scripts contain download URLs for {len(pradan_dates)} unique dates.")

    # Map dates to extracted files
    file_map: dict[str, list[Path]] = {}
    for p in all_lc_files:
        m = re.search(r"HLS_(\d{8})_", str(p))
        if not m:
            m = re.search(r"(20\d{6})", str(p))
        dt_str = m.group(1) if m else p.parent.name
        file_map.setdefault(str(dt_str), []).append(p)

    # Pre-load HEL1OS cache
    hel1os_cache: dict[Path, pd.DataFrame] = {}
    print(f"[INFO] Caching extracted HEL1OS files...")
    for p in all_lc_files:
        try:
            hel1os_cache[p] = load_hel1os_light_curve(p, energy_band="wide")
        except Exception as err:
            pass

    channels = ["cdte1", "cdte2", "czt1", "czt2"]
    min_valid_samples = 2500

    analysis_records = []

    valid_count = 0
    missing_hel1os_file_count = 0
    missing_channels_count = 0
    insufficient_samples_count = 0
    nan_values_count = 0

    for idx, row in df_flares.iterrows():
        date_str = str(row["date"])
        peak_t = float(row["peak_time"])
        start_t = row.get("start_time", peak_t - 300)
        end_t = row.get("end_time", peak_t + 300)
        goes_cls = str(row.get("goes_class", "")).strip()

        matching_files = file_map.get(date_str, [])

        is_valid = True
        missing_reason = ""
        missing_channels = []
        exist_on_pradan = date_str in pradan_dates

        if not matching_files:
            is_valid = False
            missing_reason = "No extracted HEL1OS files on disk for date"
            missing_channels = ["cdte1", "cdte2", "czt1", "czt2"]
            missing_hel1os_file_count += 1
        else:
            # Check individual channels
            target_grid = np.arange(peak_t - 3599.0, peak_t + 1.0, 1.0)
            ch_tensors = []
            
            for ch in channels:
                ch_files = [f for f in matching_files if f"lightcurve_{ch}.fits" in f.name]
                ch_dfs = [hel1os_cache[f] for f in ch_files if f in hel1os_cache]

                if not ch_dfs:
                    missing_channels.append(ch)
                else:
                    combined_ch = pd.concat(ch_dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
                    win_mask = (combined_ch["time"] >= (peak_t - 3605.0)) & (combined_ch["time"] <= (peak_t + 5.0))
                    sub_df = combined_ch[win_mask].drop_duplicates(subset=["time"]).reset_index(drop=True)

                    if len(sub_df) < min_valid_samples:
                        # Insufficient lookback samples in FITS file
                        missing_channels.append(f"{ch} (gap/short lookback: {len(sub_df)} samples)")
                    else:
                        interp_vals = np.interp(target_grid, sub_df["time"].to_numpy(), sub_df["counts"].to_numpy(), left=np.nan, right=np.nan)
                        s_vals = pd.Series(interp_vals).ffill().bfill()
                        if s_vals.isna().any():
                            missing_channels.append(f"{ch} (NaN values after interpolation)")

            if missing_channels:
                is_valid = False
                # Determine primary reason
                if len(missing_channels) == 4 and all("gap/short" in c for c in missing_channels):
                    missing_reason = "Insufficient 60-min pre-flare lookback window (<2500 samples in window)"
                    insufficient_samples_count += 1
                elif any("lightcurve" in c or c in channels for c in missing_channels):
                    missing_reason = "Missing detector channel FITS files"
                    missing_channels_count += 1
                else:
                    missing_reason = "Data gaps / NaNs in pre-flare sequence"
                    nan_values_count += 1

        if is_valid:
            valid_count += 1
        else:
            rec_status = "High (Download script exists on PRADAN)" if exist_on_pradan else "Low/Unknown (Not in current PRADAN script)"
            if "Insufficient" in missing_reason:
                rec_status = "Unrecoverable (Orbit gap/Satellite off-pointing during pre-flare window)"

            analysis_records.append({
                "flare_id": row.get("flare_id", idx),
                "observation_date": date_str,
                "peak_time_sec": peak_t,
                "start_time_sec": start_t,
                "end_time_sec": end_t,
                "goes_class": goes_cls,
                "missing_file_reason": missing_reason,
                "missing_detector_channels": ", ".join(missing_channels),
                "exist_on_pradan": exist_on_pradan,
                "pradan_url_count": len(date_to_urls.get(date_str, [])),
                "recoverability_status": rec_status
            })

    out_df = pd.DataFrame(analysis_records)
    out_df.to_csv(RECOVERABILITY_CSV, index=False)
    print(f"[SUCCESS] Analyzed {len(out_df)} excluded events out of {len(df_flares)} total catalog events.")
    print(f"Valid Sequences: {valid_count}")
    print(f"Missing HEL1OS Files on Disk: {missing_hel1os_file_count}")
    print(f"Missing Detector Channels: {missing_channels_count}")
    print(f"Insufficient Pre-flare Samples (<2500s): {insufficient_samples_count}")
    print(f"Saved analysis CSV to: {RECOVERABILITY_CSV}")

    return out_df, valid_count, missing_hel1os_file_count, insufficient_samples_count, len(df_flares)


if __name__ == "__main__":
    analyze_excluded_events()
