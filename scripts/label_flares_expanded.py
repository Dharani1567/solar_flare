"""Expanded GOES Catalog Labeling & Temporal Alignment Module (Fast Range Queries).

Queries NOAA GOES solar flare catalog in month chunks via SunPy Fido,
matches peak times within +-1800s tolerance, assigns GOES class (B, C, M, X), and exports
`data/solexs/flare_catalog/flare_events_labeled_expanded.csv`.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sunpy.net import Fido, attrs as a

from utils import PROJECT_ROOT

CATALOG_DIR = PROJECT_ROOT / "data" / "solexs" / "flare_catalog"
EXPANDED_MERGED_CSV = CATALOG_DIR / "flare_events_merged_expanded.csv"
EXPANDED_LABELED_CSV = CATALOG_DIR / "flare_events_labeled_expanded.csv"


def fetch_goes_range(start_str: str, end_str: str) -> list[dict]:
    """Fetch GOES flare records for a date range."""
    records = []
    try:
        res = Fido.search(a.Time(start_str, end_str), a.hek.FL)
        if res and len(res) > 0 and len(res[0]) > 0:
            tbl = res[0].to_pandas()
            for _, row in tbl.iterrows():
                g_cls = str(row.get("fl_goescls", "")).strip()
                p_time_str = str(row.get("event_peaktime", ""))
                if not g_cls or not p_time_str or p_time_str == "None":
                    continue
                try:
                    p_dt = pd.to_datetime(p_time_str)
                    p_ts = p_dt.timestamp()
                    d_str = p_dt.strftime("%Y%m%d")
                    records.append({
                        "goes_date": d_str,
                        "goes_class": g_cls,
                        "goes_peak_time_dt": p_dt,
                        "goes_peak_time": p_ts,
                    })
                except Exception:
                    pass
    except Exception as err:
        print(f"[WARNING] Range search error ({start_str} to {end_str}): {err}")
    return records


def fetch_goes_flares_fast(dates: list[str]) -> pd.DataFrame:
    """Fetch GOES records efficiently using month chunks."""
    min_date_str = min(dates)
    max_date_str = max(dates)

    dt_start = datetime.datetime.strptime(min_date_str, "%Y%m%d")
    dt_end = datetime.datetime.strptime(max_date_str, "%Y%m%d") + datetime.timedelta(days=1)

    print(f"[INFO] Fetching GOES flare catalog from {dt_start.strftime('%Y-%m-%d')} to {dt_end.strftime('%Y-%m-%d')}...")

    records = fetch_goes_range(dt_start.strftime("%Y-%m-%d 00:00:00"), dt_end.strftime("%Y-%m-%d 00:00:00"))

    if not records:
        print("[WARNING] Primary search yielded 0 records, searching per year fallback...")
        records.extend(fetch_goes_range("2024-01-01 00:00:00", "2024-12-31 23:59:59"))
        records.extend(fetch_goes_range("2026-01-01 00:00:00", "2026-08-25 23:59:59"))

    if not records:
        print("[WARNING] Fallback yielded 0 records.")
        return pd.DataFrame()

    df_goes = pd.DataFrame(records).drop_duplicates(subset=["goes_class", "goes_peak_time"]).reset_index(drop=True)
    print(f"[SUCCESS] Downloaded {len(df_goes)} GOES flare records across specified range.")
    return df_goes


def label_expanded_flares(merged_csv: Path | None = None) -> pd.DataFrame:
    """Perform peak-to-peak temporal matching between SoLEXS flares and GOES catalog."""
    csv_file = merged_csv or EXPANDED_MERGED_CSV
    if not csv_file.exists():
        raise FileNotFoundError(f"Merged catalog not found at {csv_file}")

    df_solexs = pd.read_csv(csv_file)
    unique_dates = sorted(df_solexs["date"].astype(str).unique())

    df_goes = fetch_goes_flares_fast(unique_dates)

    labeled_rows = []

    for idx, row in df_solexs.iterrows():
        d_str = str(row["date"])
        t_peak = float(row["peak_time"])

        row_dict = row.to_dict()
        row_dict["goes_class"] = "UNMATCHED"
        row_dict["goes_peak_time"] = np.nan
        row_dict["match_time_difference_sec"] = np.nan
        row_dict["binary_major_flare"] = 0

        if not df_goes.empty:
            sub_goes = df_goes[df_goes["goes_date"] == d_str]
            if not sub_goes.empty:
                diffs = np.abs(sub_goes["goes_peak_time"].values - t_peak)
                min_idx = np.argmin(diffs)
                min_diff = diffs[min_idx]

                if min_diff <= 1800.0:
                    matched_row = sub_goes.iloc[min_idx]
                    g_cls = str(matched_row["goes_class"]).upper()
                    row_dict["goes_class"] = g_cls
                    row_dict["goes_peak_time"] = float(matched_row["goes_peak_time"])
                    row_dict["match_time_difference_sec"] = float(min_diff)
                    if g_cls.startswith("M") or g_cls.startswith("X"):
                        row_dict["binary_major_flare"] = 1

        labeled_rows.append(row_dict)

    df_labeled = pd.DataFrame(labeled_rows)
    df_labeled.to_csv(EXPANDED_LABELED_CSV, index=False)

    pos_count = int(df_labeled["binary_major_flare"].sum())
    neg_count = len(df_labeled) - pos_count

    print(f"[SUCCESS] Exported expanded labeled catalog to: {EXPANDED_LABELED_CSV}")
    print(f"[SUMMARY] Total Flares: {len(df_labeled)} | Major M/X: {pos_count} ({pos_count/len(df_labeled)*100:.1f}%) | Minor B/C/UNMATCHED: {neg_count}")
    return df_labeled


if __name__ == "__main__":
    label_expanded_flares()
