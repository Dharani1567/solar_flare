"""Comprehensive Scientific Validation Script for SoLEXS Solar Flare Catalog."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "solexs" / "flare_catalog" / "flare_events.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOT_OUTPUT = RESULTS_DIR / "catalog_validation_plots.png"


def run_catalog_validation() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}")

    df = pd.read_csv(CATALOG_PATH)
    results = {}

    print("=" * 70)
    print("SCIENTIFIC VALIDATION REPORT: ADITYA-L1 SOLEXS FLARE CATALOG")
    print("=" * 70)

    # -------------------------------------------------------------
    # TASK 1: Analysis of Old Counts (57, 319, 389, 6786, 7760)
    # -------------------------------------------------------------
    old_sum = 57 + 319 + 389 + 6786 + 7760
    new_5day_count = df[df["date"].astype(str).isin(["20260813", "20260815", "20260817", "20260819", "20260820"])].shape[0]

    print("\n--- TASK 1: DIAGNOSIS OF OLD HIGH COUNTS ---")
    print(f"Sum of old counts (Aug 13, 15, 17, 19, 20): {old_sum:,} raw points")
    print(f"Current physics-grouped flare events (same 5 days): {new_5day_count} events")
    print(f"Reduction Ratio: {old_sum / max(new_5day_count, 1):.1f}x reduction")
    print("Conclusion: The old counts were RAW 1-SECOND POISSON NOISE PEAKS generated")
    print("by find_peaks(prominence=15, distance=1). On bright active-Sun days (Aug 19 & 20),")
    print("the background count rate rose above 15, causing thousands of consecutive 1-sec")
    print("photon fluctuations to be logged as separate 'flares'. The new pipeline groups")
    print("contiguous 60s+ signals into physical flare events.")

    # -------------------------------------------------------------
    # TASK 2: Basic Catalog Structure
    # -------------------------------------------------------------
    print("\n--- TASK 2: CATALOG SHAPE & DAILY EVENT DISTRIBUTION ---")
    print(f"Catalog Shape (Rows, Columns): {df.shape}")
    daily_counts = df.groupby("date").size()
    print("\nDaily Event Count Statistics (catalog.groupby('date').size().describe()):")
    print(daily_counts.describe())

    # -------------------------------------------------------------
    # TASK 3: Duplicate Flare Event Check
    # -------------------------------------------------------------
    print("\n--- TASK 3: DUPLICATE FLARE EVENT CHECK ---")
    exact_dups = df.duplicated(subset=["source_file", "start_time", "peak_time"]).sum()
    print(f"Exact Duplicate Entries (same source_file, start_time, peak_time): {exact_dups}")

    # Check overlapping flare time windows in same file
    overlapping_events = 0
    df_sorted = df.sort_values(["source_file", "start_time"]).reset_index(drop=True)
    for i in range(len(df_sorted) - 1):
        if df_sorted.loc[i, "source_file"] == df_sorted.loc[i + 1, "source_file"]:
            end1 = df_sorted.loc[i, "end_time"]
            start2 = df_sorted.loc[i + 1, "start_time"]
            if start2 < end1:
                overlapping_events += 1

    print(f"Overlapping Time Window Events: {overlapping_events}")

    # -------------------------------------------------------------
    # TASK 4: Multi-Peak & Close Event Separation Check
    # -------------------------------------------------------------
    print("\n--- TASK 4: CLOSELY SPACED CANDIDATE FLARES (<60s GAP) ---")
    close_gaps = 0
    for i in range(len(df_sorted) - 1):
        if df_sorted.loc[i, "source_file"] == df_sorted.loc[i + 1, "source_file"]:
            end1 = df_sorted.loc[i, "end_time"]
            start2 = df_sorted.loc[i + 1, "start_time"]
            gap = start2 - end1
            if 0 <= gap < 60:
                close_gaps += 1

    print(f"Events separated by <60 seconds: {close_gaps} ({close_gaps / max(len(df), 1) * 100:.1f}% of catalog)")

    # -------------------------------------------------------------
    # TASK 5: Duration Verification
    # -------------------------------------------------------------
    print("\n--- TASK 5: DURATION CALCULATION VERIFICATION ---")
    calculated_dur = df_sorted["end_time"] - df_sorted["start_time"]
    dur_mismatch = np.abs(calculated_dur - df_sorted["duration_sec"]).max()
    print(f"Max difference between (end_time - start_time) and duration_sec: {dur_mismatch:.6f} seconds")
    print("\nDuration Distribution (seconds):")
    print(df["duration_sec"].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

    short_flares = (df["duration_sec"] < 60).sum()
    long_flares = (df["duration_sec"] > 3600).sum()
    print(f"Flares <60s: {short_flares} | Flares >3600s (1h+): {long_flares}")

    # -------------------------------------------------------------
    # TASK 6: SNR Verification
    # -------------------------------------------------------------
    print("\n--- TASK 6: SNR CALCULATION VERIFICATION ---")
    expected_snr = df["net_peak_counts"] / np.sqrt(np.maximum(df["background_counts"], 1.0))
    snr_diff = np.abs(expected_snr - df["snr"]).max()
    print(f"Max difference between (net_peak / sqrt(background)) and logged snr: {snr_diff:.6f}")
    print("\nSNR Distribution:")
    print(df["snr"].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

    # -------------------------------------------------------------
    # TASK 7: Top 20 Strongest Flares
    # -------------------------------------------------------------
    print("\n--- TASK 7: TOP 20 FLARES BY METRIC ---")

    top_counts = df.nlargest(20, "peak_counts")[["date", "source_file", "peak_counts", "net_peak_counts", "snr", "duration_sec"]]
    print("\nTop 20 Flares by Peak Counts:")
    print(top_counts.to_string(index=False))

    top_snr = df.nlargest(20, "snr")[["date", "source_file", "snr", "peak_counts", "background_counts", "duration_sec"]]
    print("\nTop 20 Flares by SNR:")
    print(top_snr.to_string(index=False))

    top_duration = df.nlargest(20, "duration_sec")[["date", "source_file", "duration_sec", "peak_counts", "snr"]]
    print("\nTop 20 Flares by Duration:")
    print(top_duration.to_string(index=False))

    # -------------------------------------------------------------
    # TASK 8: Validation Plots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Aditya-L1 SoLEXS Solar Flare Catalog Scientific Validation", fontsize=16, fontweight="bold")

    # Plot 1: Daily Event Count
    ax1 = axes[0, 0]
    daily_counts.plot(kind="line", ax=ax1, color="#1f77b4", linewidth=1.5, marker="o", markersize=3)
    ax1.set_title("Flares Detected Per Day (117 Days)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Flare Count")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Duration Histogram
    ax2 = axes[0, 1]
    ax2.hist(df["duration_sec"] / 60.0, bins=40, color="#2ca02c", edgecolor="black", alpha=0.7)
    ax2.set_title("Flare Duration Distribution (Minutes)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Duration (Minutes)")
    ax2.set_ylabel("Frequency")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Plot 3: SNR Histogram
    ax3 = axes[1, 0]
    ax3.hist(df["snr"], bins=40, color="#ff7f0e", edgecolor="black", alpha=0.7)
    ax3.set_title("Signal-to-Noise Ratio (SNR) Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("SNR")
    ax3.set_ylabel("Frequency")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Peak Counts (Log N - Log S Flare Power Law)
    ax4 = axes[1, 1]
    counts_sorted = np.sort(df["net_peak_counts"].values)[::-1]
    ranks = np.arange(1, len(counts_sorted) + 1)
    ax4.loglog(counts_sorted, ranks, marker=".", linestyle="none", color="#d62728", alpha=0.7)
    ax4.set_title("Log(N) - Log(S) Frequency Distribution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Net Peak Counts (S)")
    ax4.set_ylabel("Cumulative Event Count N(>S)")
    ax4.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=300)
    plt.close()
    print(f"\nValidation plots successfully saved to: {PLOT_OUTPUT}")

    # -------------------------------------------------------------
    # TASK 9: Flag Suspicious Days
    # -------------------------------------------------------------
    print("\n--- TASK 9: SUSPICIOUS DAYS ANOMALY DETECTION ---")
    mean_daily = daily_counts.mean()
    std_daily = daily_counts.std()

    high_days = daily_counts[daily_counts > mean_daily + 2 * std_daily]
    low_days = daily_counts[daily_counts <= 1]

    daily_avg_dur = df.groupby("date")["duration_sec"].mean()
    long_dur_days = daily_avg_dur[daily_avg_dur > 1500]

    daily_avg_snr = df.groupby("date")["snr"].mean()
    high_snr_days = daily_avg_snr[daily_avg_snr > 80]

    print(f"\nHigh Flare Count Days (> {mean_daily + 2*std_daily:.1f} flares/day):")
    print(high_days if not high_days.empty else "None")

    print(f"\nLow Flare Count Days (<= 1 flare/day):")
    print(f"{len(low_days)} days flagged (sample: {low_days.head(5).index.tolist()})")

    print(f"\nHigh Average Duration Days (> 1500s / 25min avg):")
    print(long_dur_days if not long_dur_days.empty else "None")

    print(f"\nHigh Average SNR Days (> 80 avg SNR):")
    print(high_snr_days if not high_snr_days.empty else "None")

    print("=" * 70)
    return results


if __name__ == "__main__":
    run_ingestion_pipeline = run_catalog_validation()
