"""GOES Flare Labeling Pipeline for Aditya-L1 SoLEXS Catalog.

This module downloads the GOES flare catalog using SunPy Fido (HEK service),
temporally matches each SoLEXS flare event to its nearest GOES flare within a
configurable time tolerance, adds GOES metadata fields, reports matching statistics,
and generates GOES class distribution diagnostic plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sunpy.net import Fido, attrs as a

from utils import (
    CATALOG_DIR,
    GOES_CATALOG_FILE,
    GOES_PLOT_OUTPUT,
    LABELED_CATALOG_FILE,
    MERGED_CATALOG_FILE,
    PROJECT_ROOT,
    RESULTS_DIR,
)


def download_goes_catalog(
    solexs_df: pd.DataFrame,
    force_download: bool = False,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """Download GOES flare catalog via SunPy Fido (HEK) for date ranges present in SoLEXS catalog.

    Caches results locally to avoid redundant HTTP downloads.
    """
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    if GOES_CATALOG_FILE.exists() and not force_download:
        print(f"[INFO] Loading cached GOES flare catalog from: {GOES_CATALOG_FILE}")
        goes_df = pd.read_csv(GOES_CATALOG_FILE)
        # Parse timestamp column
        goes_df["peak_time_dt"] = pd.to_datetime(goes_df["event_peaktime"], utc=True)
        goes_df["peak_time_sec"] = (goes_df["peak_time_dt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        return goes_df

    print("[INFO] Fetching GOES flare catalog using SunPy Fido (HEK)...")

    # Determine date clusters in SoLEXS dataset to optimize network requests
    dates = pd.to_datetime(solexs_df["date"].astype(str), format="%Y%m%d").sort_values()
    gaps = dates.diff() > pd.Timedelta(days=5)
    group_ids = gaps.cumsum()

    all_hek_rows = []

    for _, group in dates.groupby(group_ids):
        # Buffer range by 1 day on each side to ensure flares at date boundaries are captured
        start_dt = group.min() - pd.Timedelta(days=1)
        end_dt = group.max() + pd.Timedelta(days=1)

        cur_start = start_dt
        while cur_start < end_dt:
            cur_end = min(cur_start + pd.Timedelta(days=chunk_days), end_dt)
            t_start_str = cur_start.strftime("%Y-%m-%d 00:00:00")
            t_end_str = cur_end.strftime("%Y-%m-%d 23:59:59")

            print(f"  -> Querying HEK for GOES flares between {t_start_str[:10]} and {t_end_str[:10]}...")
            try:
                res = Fido.search(
                    a.Time(t_start_str, t_end_str),
                    a.hek.EventType("FL"),
                    a.hek.OBS.Observatory == "GOES",
                )
                if len(res) > 0 and len(res[0]) > 0:
                    sub_df = res[0].to_pandas()
                    all_hek_rows.append(sub_df)
                    print(f"     Found {len(sub_df)} GOES flare events.")
                else:
                    print("     No GOES flare events found in this window.")
            except Exception as err:
                print(f"     [WARNING] Error querying range {t_start_str[:10]} to {t_end_str[:10]}: {err}")

            cur_start = cur_end + pd.Timedelta(seconds=1)

    if not all_hek_rows:
        raise RuntimeError("Failed to fetch any GOES flare catalog data from HEK.")

    combined_hek = pd.concat(all_hek_rows, ignore_index=True)

    # Clean & deduplicate GOES flare catalog
    # Required fields: event_peaktime, event_starttime, event_endtime, fl_goescls
    cols_to_keep = [c for c in ["event_starttime", "event_peaktime", "event_endtime", "fl_goescls", "ar_noaanum", "SOL_standard"] if c in combined_hek.columns]
    goes_df = combined_hek[cols_to_keep].dropna(subset=["event_peaktime", "fl_goescls"]).copy()

    goes_df["fl_goescls"] = goes_df["fl_goescls"].astype(str).str.strip()
    goes_df = goes_df[goes_df["fl_goescls"] != ""].copy()

    # Convert peaktime to pandas datetime UTC
    goes_df["peak_time_dt"] = pd.to_datetime(goes_df["event_peaktime"], utc=True)
    goes_df = goes_df.drop_duplicates(subset=["event_peaktime", "fl_goescls"]).sort_values("peak_time_dt").reset_index(drop=True)

    # Save to disk cache
    export_df = goes_df.drop(columns=["peak_time_dt"], errors="ignore")
    export_df.to_csv(GOES_CATALOG_FILE, index=False)
    print(f"[SUCCESS] Saved {len(goes_df)} clean GOES flare records to: {GOES_CATALOG_FILE}")

    goes_df["peak_time_sec"] = (goes_df["peak_time_dt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return goes_df


def label_solexs_flares(
    solexs_df: pd.DataFrame,
    goes_df: pd.DataFrame,
    tolerance_sec: float = 1800.0,
) -> pd.DataFrame:
    """Match each SoLEXS flare event to the nearest GOES flare within configurable time tolerance.

    Parameters
    ----------
    solexs_df : pd.DataFrame
        Merged SoLEXS flare catalog containing 'peak_time' (POSIX seconds).
    goes_df : pd.DataFrame
        GOES flare catalog containing 'peak_time_sec' and 'fl_goescls'.
    tolerance_sec : float
        Maximum allowed absolute time difference in seconds between SoLEXS peak and GOES peak.

    Returns
    -------
    pd.DataFrame
        Labeled catalog with additional columns: goes_class, goes_peak_time, match_time_difference_sec.
    """
    labeled_df = solexs_df.copy()

    goes_peak_secs = goes_df["peak_time_sec"].values
    goes_classes = goes_df["fl_goescls"].values
    goes_peak_dts = goes_df["event_peaktime"].values

    match_goes_class = []
    match_goes_peak_time = []
    match_time_diff_sec = []

    for _, row in labeled_df.iterrows():
        s_peak = float(row["peak_time"])

        if len(goes_peak_secs) == 0:
            match_goes_class.append("UNMATCHED")
            match_goes_peak_time.append(np.nan)
            match_time_diff_sec.append(np.nan)
            continue

        # Compute absolute time differences to all GOES flares
        diffs_signed = s_peak - goes_peak_secs
        abs_diffs = np.abs(diffs_signed)

        best_idx = np.argmin(abs_diffs)
        min_diff = abs_diffs[best_idx]
        signed_diff = diffs_signed[best_idx]

        if min_diff <= tolerance_sec:
            match_goes_class.append(str(goes_classes[best_idx]))
            match_goes_peak_time.append(str(goes_peak_dts[best_idx]))
            match_time_diff_sec.append(round(float(signed_diff), 2))
        else:
            match_goes_class.append("UNMATCHED")
            match_goes_peak_time.append(np.nan)
            match_time_diff_sec.append(np.nan)

    labeled_df["goes_class"] = match_goes_class
    labeled_df["goes_peak_time"] = match_goes_peak_time
    labeled_df["match_time_difference_sec"] = match_time_diff_sec

    return labeled_df


def _extract_goes_letter(cls_str: str) -> str:
    """Extract major GOES flare class letter (X, M, C, B, A, or UNMATCHED)."""
    cls_str = str(cls_str).strip().upper()
    if cls_str.startswith("X"):
        return "X"
    if cls_str.startswith("M"):
        return "M"
    if cls_str.startswith("C"):
        return "C"
    if cls_str.startswith("B"):
        return "B"
    if cls_str.startswith("A"):
        return "A"
    return "UNMATCHED"


def report_matching_summary(df: pd.DataFrame, tolerance_sec: float) -> None:
    """Print statistical summary report of temporal matching results."""
    total_events = len(df)
    matched_df = df[df["goes_class"] != "UNMATCHED"]
    unmatched_df = df[df["goes_class"] == "UNMATCHED"]

    matched_count = len(matched_df)
    unmatched_count = len(unmatched_df)
    match_pct = (matched_count / total_events) * 100 if total_events > 0 else 0.0

    # Class letter distribution
    letter_series = df["goes_class"].apply(_extract_goes_letter)
    letter_counts = letter_series.value_counts().to_dict()

    print("\n" + "=" * 70)
    print("GOES FLARE LABELING PIPELINE REPORT")
    print("=" * 70)
    print(f"Time Tolerance Window      : +/- {tolerance_sec:.1f} seconds ({tolerance_sec/60:.1f} minutes)")
    print(f"Total SoLEXS Merged Flares  : {total_events:,}")
    print(f"Matched Events              : {matched_count:,}")
    print(f"Unmatched Events            : {unmatched_count:,}")
    print(f"Match Percentage            : {match_pct:.2f}%")

    print("\n[GOES Flare Class Breakdown]")
    for letter in ["X", "M", "C", "B", "A", "UNMATCHED"]:
        cnt = letter_counts.get(letter, 0)
        pct = (cnt / total_events) * 100 if total_events > 0 else 0.0
        print(f"  Class {letter:9s} : {cnt:5d} ({pct:6.2f}%)")

    if matched_count > 0:
        time_diffs = matched_df["match_time_difference_sec"].abs()
        print("\n[Temporal Match Differences (|t_SoLEXS - t_GOES|)]")
        print(f"  Mean Difference          : {time_diffs.mean():.2f} sec ({time_diffs.mean()/60:.2f} min)")
        print(f"  Median Difference        : {time_diffs.median():.2f} sec ({time_diffs.median()/60:.2f} min)")
        print(f"  Std Deviation            : {time_diffs.std():.2f} sec")
        print(f"  Max Match Difference     : {time_diffs.max():.2f} sec ({time_diffs.max()/60:.2f} min)")

    print("=" * 70 + "\n")


def generate_distribution_plots(df: pd.DataFrame, output_path: Path) -> None:
    """Generate diagnostic plots for GOES class distribution and temporal match quality."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SoLEXS Flare GOES Labeling & Distribution Analysis", fontsize=15, fontweight="bold")

    # Plot 1: Major GOES Class Count Bar Chart
    ax1 = axes[0]
    letter_series = df["goes_class"].apply(_extract_goes_letter)
    categories = ["X", "M", "C", "B", "A", "UNMATCHED"]
    counts = [letter_series.value_counts().get(cat, 0) for cat in categories]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#7f7f7f"]

    bars = ax1.bar(categories, counts, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("GOES Class Breakdown", fontsize=12, fontweight="bold")
    ax1.set_xlabel("GOES Flare Class")
    ax1.set_ylabel("Number of SoLEXS Flares")
    ax1.grid(True, axis="y", alpha=0.3)

    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(counts) * 0.01,
                f"{cnt}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Plot 2: Time Difference Distribution Histogram for Matched Events
    ax2 = axes[1]
    matched_df = df[df["goes_class"] != "UNMATCHED"]
    if not matched_df.empty:
        diff_mins = matched_df["match_time_difference_sec"] / 60.0
        ax2.hist(diff_mins, bins=30, color="#17becf", edgecolor="black", alpha=0.75)
        ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Exact Synchrony (0 min)")
        ax2.set_title("Temporal Peak Offset (SoLEXS - GOES)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Time Difference (Minutes)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No Matched Events", ha="center", va="center", transform=ax2.transAxes)

    # Plot 3: SoLEXS Net Peak Counts vs GOES Flare Class (Box Plot)
    ax3 = axes[2]
    df_plot = df.copy()
    df_plot["major_class"] = df_plot["goes_class"].apply(_extract_goes_letter)

    valid_cats = [cat for cat in categories if cat in df_plot["major_class"].unique() and cat != "UNMATCHED"]
    if valid_cats:
        plot_data = [df_plot[df_plot["major_class"] == cat]["net_peak_counts"].dropna() for cat in valid_cats]
        ax3.boxplot(plot_data, tick_labels=valid_cats, patch_artist=True, boxprops=dict(facecolor="#aec7e8", alpha=0.7))
        ax3.set_yscale("log")
        ax3.set_title("SoLEXS Net Peak Counts by GOES Class", fontsize=12, fontweight="bold")
        ax3.set_xlabel("GOES Class")
        ax3.set_ylabel("Net Peak Counts (Log Scale)")
        ax3.grid(True, which="both", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No GOES Class Data", ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] GOES distribution plots saved to: {output_path}")


def run_pipeline(
    tolerance_sec: float = 1800.0,
    force_download: bool = False,
) -> pd.DataFrame:
    """Execute complete GOES flare labeling pipeline."""
    if not MERGED_CATALOG_FILE.exists():
        raise FileNotFoundError(f"Merged flare catalog not found at: {MERGED_CATALOG_FILE}")

    print(f"[INFO] Loading merged flare catalog from: {MERGED_CATALOG_FILE}")
    solexs_df = pd.read_csv(MERGED_CATALOG_FILE)

    # 1. Download/load GOES flare catalog
    goes_df = download_goes_catalog(solexs_df, force_download=force_download)

    # 2. Match SoLEXS flares to GOES flares
    print(f"[INFO] Temporal matching in progress with tolerance = +/- {tolerance_sec} seconds...")
    labeled_df = label_solexs_flares(solexs_df, goes_df, tolerance_sec=tolerance_sec)

    # 3. Export labeled catalog CSV
    labeled_df.to_csv(LABELED_CATALOG_FILE, index=False)
    print(f"[SUCCESS] Exported labeled catalog to: {LABELED_CATALOG_FILE}")

    # 4. Generate report & diagnostic plots
    report_matching_summary(labeled_df, tolerance_sec=tolerance_sec)
    generate_distribution_plots(labeled_df, output_path=GOES_PLOT_OUTPUT)

    return labeled_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GOES Solar Flare Labeling Pipeline for Aditya-L1 SoLEXS")
    parser.add_argument("--tolerance", type=float, default=1800.0, help="Time tolerance window in seconds (default: 1800.0s / 30m)")
    parser.add_argument("--force-download", action="store_true", help="Force fresh download of GOES flare catalog from HEK")
    args = parser.parse_args()

    run_pipeline(tolerance_sec=args.tolerance, force_download=args.force_download)
