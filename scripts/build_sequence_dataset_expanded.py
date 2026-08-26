"""Raw HEL1OS 1 Hz Sequence Dataset Generator for Expanded Dataset.

Extracts 60-minute (3,600 time steps at 1 Hz) pre-flare raw light curve time-series
across 4 detector channels (cdte1, cdte2, czt1, czt2) for all expanded catalog flares.

Outputs:
- `data/ml/X_sequences_expanded.npy` -> Shape (N, 3600, 4)
- `data/ml/y_labels_expanded.npy` -> Shape (N,)
- `data/ml/sequence_metadata_expanded.csv` -> Flare metadata matching tensor rows
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

from load_hel1os import list_hel1os_files, load_hel1os_light_curve
from utils import (
    HEL1OS_EXTRACTED_DIR,
    PROJECT_ROOT,
)

ML_DIR = PROJECT_ROOT / "data" / "ml"
EXPANDED_LABELED_CSV = PROJECT_ROOT / "data" / "solexs" / "flare_catalog" / "flare_events_labeled_expanded.csv"

X_NPY_EXPANDED = ML_DIR / "X_sequences_expanded.npy"
Y_NPY_EXPANDED = ML_DIR / "y_labels_expanded.npy"
META_CSV_EXPANDED = ML_DIR / "sequence_metadata_expanded.csv"


def build_expanded_sequence_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract raw 1 Hz 3,600-second sequence tensors for all expanded flares with HEL1OS coverage."""
    if not EXPANDED_LABELED_CSV.exists():
        raise FileNotFoundError(f"Expanded catalog not found at {EXPANDED_LABELED_CSV}")

    df_flares = pd.read_csv(EXPANDED_LABELED_CSV)
    all_lc_files = list_hel1os_files(HEL1OS_EXTRACTED_DIR)

    file_map: dict[str, list[Path]] = {}
    for p in all_lc_files:
        m = re.search(r"HLS_(\d{8})_", str(p))
        if not m:
            m = re.search(r"(20\d{6})", str(p))
        dt_str = m.group(1) if m else p.parent.name
        file_map.setdefault(dt_str, []).append(p)

    hel1os_cache: dict[Path, pd.DataFrame] = {}
    print(f"[INFO] Pre-loading HEL1OS cache for {len(all_lc_files)} extracted FITS light curve files...")
    for p in all_lc_files:
        try:
            hel1os_cache[p] = load_hel1os_light_curve(p, energy_band="wide")
        except Exception as err:
            pass

    print(f"[INFO] Extracting 60-minute sequence tensors for {len(df_flares)} expanded flares...")

    channels = ["cdte1", "cdte2", "czt1", "czt2"]
    sequences_list = []
    labels_list = []
    meta_rows = []

    min_valid_samples = 2500

    for idx, row in df_flares.iterrows():
        date_str = str(row["date"])
        t_peak = float(row["peak_time"])

        g_cls = str(row.get("goes_class", "")).strip().upper()
        label = 1 if (g_cls.startswith("M") or g_cls.startswith("X")) else 0

        matching_files = file_map.get(date_str, [])
        if not matching_files:
            continue

        target_grid = np.arange(t_peak - 3599.0, t_peak + 1.0, 1.0)
        assert len(target_grid) == 3600

        ch_tensors = []
        is_flare_valid = True

        for ch in channels:
            ch_files = [f for f in matching_files if f"lightcurve_{ch}.fits" in f.name]
            ch_dfs = [hel1os_cache[f] for f in ch_files if f in hel1os_cache]

            if not ch_dfs:
                is_flare_valid = False
                break

            combined_ch = pd.concat(ch_dfs, ignore_index=True).sort_values("time").reset_index(drop=True)

            win_mask = (combined_ch["time"] >= (t_peak - 3605.0)) & (combined_ch["time"] <= (t_peak + 5.0))
            sub_df = combined_ch[win_mask].drop_duplicates(subset=["time"]).reset_index(drop=True)

            if len(sub_df) < min_valid_samples:
                is_flare_valid = False
                break

            interp_vals = np.interp(target_grid, sub_df["time"].to_numpy(), sub_df["counts"].to_numpy(), left=np.nan, right=np.nan)
            s_vals = pd.Series(interp_vals).ffill().bfill()

            if s_vals.isna().any():
                is_flare_valid = False
                break

            ch_tensors.append(s_vals.to_numpy(dtype=np.float32))

        if is_flare_valid and len(ch_tensors) == 4:
            seq_matrix = np.column_stack(ch_tensors)
            assert seq_matrix.shape == (3600, 4)
            assert not np.isnan(seq_matrix).any()

            sequences_list.append(seq_matrix)
            labels_list.append(label)

            meta_dict = row.to_dict()
            meta_dict["binary_major_flare"] = label
            meta_dict["seq_idx"] = len(sequences_list) - 1
            meta_rows.append(meta_dict)

    if not sequences_list:
        raise ValueError("No valid sequence samples could be extracted for expanded dataset!")

    X = np.array(sequences_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)

    ML_DIR.mkdir(parents=True, exist_ok=True)
    np.save(X_NPY_EXPANDED, X)
    np.save(Y_NPY_EXPANDED, y)
    meta_df.to_csv(META_CSV_EXPANDED, index=False)

    print(f"[SUCCESS] Saved expanded X tensor to: {X_NPY_EXPANDED} (Shape: {X.shape})")
    print(f"[SUCCESS] Saved expanded y tensor to: {Y_NPY_EXPANDED} (Shape: {y.shape})")
    print(f"[SUCCESS] Saved expanded metadata to: {META_CSV_EXPANDED}")

    return X, y, meta_df


if __name__ == "__main__":
    build_expanded_sequence_dataset()
