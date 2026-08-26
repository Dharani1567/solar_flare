"""Build the master SoLEXS flare catalog from extracted light curves."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from detect_flares import detect_flare_events
from load_solexs import read_light_curve
from utils import CATALOG_DIR, ensure_output_directories, observation_date, solexs_light_curves


def build_catalog(
    count_column: str = "counts",
    output_path: Path | None = None,
    sigma_threshold: float = 4.5,
    min_duration_sec: float = 60.0,
) -> pd.DataFrame:
    """Detect flare events in every extracted light curve and write master CSV.

    `count_column` must match the column name in the light curve files.
    """
    events: list[pd.DataFrame] = []
    for file_path in solexs_light_curves():
        print(f"Processing: {file_path.name}")

        frame = read_light_curve(file_path)

        if count_column not in frame.columns:
            # Fallback column search if needed
            possible_cols = [c for c in frame.columns if "count" in c or "rate" in c or "flux" in c]
            selected_col = possible_cols[0] if possible_cols else frame.columns[1]
        else:
            selected_col = count_column

        time_vals = frame["time"].astype(float).to_numpy()
        count_vals = frame[selected_col].astype(float).to_numpy()

        detected = detect_flare_events(
            time_array=time_vals,
            count_array=count_vals,
            sigma_threshold=sigma_threshold,
            min_duration_sec=min_duration_sec,
        )

        if not detected.empty:
            detected.insert(0, "date", observation_date(file_path))
            detected.insert(1, "source_file", file_path.name)
            events.append(detected)

    catalog_columns = [
        "date",
        "source_file",
        "start_time",
        "peak_time",
        "end_time",
        "duration_sec",
        "peak_counts",
        "background_counts",
        "net_peak_counts",
        "snr",
    ]

    catalog = (
        pd.concat(events, ignore_index=True)
        if events
        else pd.DataFrame(columns=catalog_columns)
    )

    ensure_output_directories()
    output_path = output_path or CATALOG_DIR / "flare_events.csv"
    catalog.to_csv(output_path, index=False)
    print(f"Catalog saved to {output_path} with {len(catalog)} events.")
    return catalog


if __name__ == "__main__":
    files = solexs_light_curves()
    print(f"Found {len(files)} light curve files.")

    catalog = build_catalog("counts")
    print("\nSummary of build:")
    print(f"Total detected events: {len(catalog)}")