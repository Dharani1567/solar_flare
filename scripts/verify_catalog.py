"""Verify and analyze the generated solar flare catalog."""

from __future__ import annotations

import pandas as pd
from utils import CATALOG_DIR


def analyze_catalog() -> None:
    catalog_path = CATALOG_DIR / "flare_events.csv"
    if not catalog_path.exists():
        print(f"Catalog not found at {catalog_path}")
        return

    df = pd.read_csv(catalog_path)
    print("=" * 60)
    print("SOLAR FLARE CATALOG DIAGNOSTICS & QUALITY SUMMARY")
    print("=" * 60)
    print(f"Total Detected Flare Events: {len(df)}")
    if df.empty:
        return

    print("\n[Event Columns]")
    print(df.columns.tolist())

    print("\n[Summary Statistics]")
    print(df[["duration_sec", "peak_counts", "background_counts", "net_peak_counts", "snr"]].describe())

    print("\n[Events Per Date]")
    print(df.groupby("date").size())

    print("\n[First 10 Events sample]")
    print(df[["date", "source_file", "start_time", "peak_time", "end_time", "duration_sec", "net_peak_counts", "snr"]].head(10))
    print("=" * 60)


if __name__ == "__main__":
    analyze_catalog()
