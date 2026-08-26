"""Production-grade Flare Merger Module for Aditya-L1 SoLEXS catalog."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import CATALOG_DIR, PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
INPUT_CATALOG = CATALOG_DIR / "flare_events.csv"
OUTPUT_CATALOG = CATALOG_DIR / "flare_events_merged.csv"
PLOT_OUTPUT = RESULTS_DIR / "merger_diagnostics_plots.png"


def _collapse_group(group: list[dict]) -> dict:
    """Collapse a list of sub-event dicts into a single merged flare event."""
    first = group[0]
    last = group[-1]

    # Identify sub-event with maximum peak counts
    max_peak_event = max(group, key=lambda x: x["peak_counts"])

    start_time = float(first["start_time"])
    end_time = float(last["end_time"])
    duration_sec = end_time - start_time
    peak_counts = float(max_peak_event["peak_counts"])
    background_counts = float(max_peak_event["background_counts"])
    net_peak_counts = peak_counts - background_counts
    max_snr = float(max(x["snr"] for x in group))

    return {
        "date": first["date"],
        "source_file": first["source_file"],
        "start_time": start_time,
        "peak_time": float(max_peak_event["peak_time"]),
        "end_time": end_time,
        "duration_sec": duration_sec,
        "peak_counts": peak_counts,
        "background_counts": background_counts,
        "net_peak_counts": net_peak_counts,
        "snr": max_snr,
        "event_count_merged": len(group),
    }


def merge_flare_catalog(df: pd.DataFrame, max_gap_sec: float = 120.0) -> pd.DataFrame:
    """Merge adjacent flare events within max_gap_sec from the same observation file."""
    if df.empty:
        return df.copy()

    df_sorted = df.sort_values(["source_file", "start_time"]).reset_index(drop=True)

    merged_rows = []
    current_group = [df_sorted.iloc[0].to_dict()]

    for i in range(1, len(df_sorted)):
        prev_event = current_group[-1]
        curr_event = df_sorted.iloc[i].to_dict()

        same_file = curr_event["source_file"] == prev_event["source_file"]
        gap = curr_event["start_time"] - prev_event["end_time"]

        if same_file and gap <= max_gap_sec:
            current_group.append(curr_event)
        else:
            merged_rows.append(_collapse_group(current_group))
            current_group = [curr_event]

    if current_group:
        merged_rows.append(_collapse_group(current_group))

    return pd.DataFrame(merged_rows)


def run_merger_diagnostics(max_gap_sec: float = 120.0) -> None:
    """Run full merger execution, print comparison stats, and save diagnostic plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CATALOG.exists():
        raise FileNotFoundError(f"Input catalog not found at {INPUT_CATALOG}")

    raw_df = pd.read_csv(INPUT_CATALOG)
    raw_df_sorted = raw_df.sort_values(["source_file", "start_time"]).reset_index(drop=True)

    # 1. Compute inter-event gap times (seconds) for consecutive flares in same file
    gaps = []
    for i in range(len(raw_df_sorted) - 1):
        if raw_df_sorted.loc[i, "source_file"] == raw_df_sorted.loc[i + 1, "source_file"]:
            gap = raw_df_sorted.loc[i + 1, "start_time"] - raw_df_sorted.loc[i, "end_time"]
            gaps.append(gap)

    gaps_arr = np.array(gaps)
    gaps_below_threshold = (gaps_arr <= max_gap_sec).sum()

    # 2. Perform merging
    merged_df = merge_flare_catalog(raw_df, max_gap_sec=max_gap_sec)

    # 3. Save output catalog
    merged_df.to_csv(OUTPUT_CATALOG, index=False)

    # 4. Compare statistics
    orig_count = len(raw_df)
    merged_count = len(merged_df)
    multi_peak_events = (merged_df["event_count_merged"] > 1).sum()
    max_sub_events = merged_df["event_count_merged"].max()
    reduction_pct = ((orig_count - merged_count) / orig_count) * 100

    print("=" * 70)
    print("SOLAR FLARE EVENT MERGER DIAGNOSTICS REPORT")
    print("=" * 70)
    print(f"Max Merger Gap Threshold : {max_gap_sec:.1f} seconds")
    print(f"Original Event Count     : {orig_count:,}")
    print(f"Merged Event Count       : {merged_count:,}")
    print(f"Total Events Combined    : {orig_count - merged_count:,}")
    print(f"Reduction Percentage     : {reduction_pct:.2f}%")
    print(f"Multi-Peak Merged Events : {multi_peak_events:,} ({multi_peak_events/merged_count*100:.1f}% of merged catalog)")
    print(f"Max Sub-Events in 1 Flare: {max_sub_events} sub-peaks")
    print(f"Output Saved To          : {OUTPUT_CATALOG}")

    print("\n[Merged Event Count Statistics]")
    print(merged_df["event_count_merged"].describe())

    print("\n[Sample Merged Multi-Peak Events]")
    multi_sample = merged_df[merged_df["event_count_merged"] > 1][
        ["date", "source_file", "event_count_merged", "duration_sec", "peak_counts", "snr"]
    ].head(10)
    print(multi_sample.to_string(index=False))

    # 5. Produce Diagnostic Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"SoLEXS Flare Event Merger Diagnostics (Threshold: {max_gap_sec:.0f}s)", fontsize=14, fontweight="bold")

    # Plot A: Inter-Event Gap Histogram
    ax1 = axes[0]
    gaps_filtered = gaps_arr[gaps_arr <= 1800]  # Filter gaps up to 30 mins for plotting
    ax1.hist(gaps_filtered / 60.0, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax1.axvline(max_gap_sec / 60.0, color="red", linestyle="--", linewidth=2, label=f"Merger Cutoff ({max_gap_sec/60:.1f} min)")
    ax1.set_title("Inter-Event Gap Distribution (Gaps <= 30 mins)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Gap Time Between Consecutive Flares (Minutes)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot B: Before vs After Event Count per Date Comparison
    ax2 = axes[1]
    raw_daily = raw_df.groupby("date").size()
    merged_daily = merged_df.groupby("date").size()

    dates = raw_daily.index.astype(str)
    x = np.arange(len(dates))
    width = 0.4

    ax2.bar(x - width / 2, raw_daily.values, width, label="Original Events", color="#ff7f0e", alpha=0.8)
    ax2.bar(x + width / 2, merged_daily.values, width, label="Merged Events", color="#2ca02c", alpha=0.8)
    ax2.set_title("Daily Flare Count: Original vs Merged Catalog", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Observation Date (Sample)")
    ax2.set_ylabel("Event Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Set limited ticks for clean presentation
    tick_step = max(len(dates) // 10, 1)
    ax2.set_xticks(x[::tick_step])
    ax2.set_xticklabels(dates[::tick_step], rotation=45)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=300)
    plt.close()
    print(f"\nMerger diagnostic plot successfully saved to: {PLOT_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    run_merger_diagnostics(max_gap_sec=120.0)
