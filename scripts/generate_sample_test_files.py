"""Generate Sample Test Input Files (.npy and .csv) for Testing Model Inference.

Creates:
- data/test_samples/sample_major_flare_event.npy (3600, 4)
- data/test_samples/sample_major_flare_event.csv (3600 rows x 4 cols)
- data/test_samples/sample_quiet_sun_event.npy (3600, 4)
- data/test_samples/sample_quiet_sun_event.csv (3600 rows x 4 cols)
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml"
TEST_DIR = PROJECT_ROOT / "data" / "test_samples"

X_PATH = DATA_DIR / "X_sequences_expanded.npy"
Y_PATH = DATA_DIR / "y_labels_expanded.npy"

TEST_DIR.mkdir(parents=True, exist_ok=True)

if X_PATH.exists() and Y_PATH.exists():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    major_idx = np.where(y == 1)[0][0]
    quiet_idx = np.where(y == 0)[0][0]

    major_sample = X[major_idx]  # (3600, 4)
    quiet_sample = X[quiet_idx]  # (3600, 4)

    # Save .npy files
    np.save(TEST_DIR / "sample_major_flare_event.npy", major_sample)
    np.save(TEST_DIR / "sample_quiet_sun_event.npy", quiet_sample)

    # Save .csv files
    df_major = pd.DataFrame(major_sample, columns=["cdte1", "cdte2", "czt1", "czt2"])
    df_quiet = pd.DataFrame(quiet_sample, columns=["cdte1", "cdte2", "czt1", "czt2"])

    df_major.to_csv(TEST_DIR / "sample_major_flare_event.csv", index=False)
    df_quiet.to_csv(TEST_DIR / "sample_quiet_sun_event.csv", index=False)

    print(f"[SUCCESS] Exported Sample Test Files to: {TEST_DIR}")
    print(f"  1. Major Flare NPY : sample_major_flare_event.npy {major_sample.shape}")
    print(f"  2. Major Flare CSV : sample_major_flare_event.csv {df_major.shape}")
    print(f"  3. Quiet Sun NPY   : sample_quiet_sun_event.npy {quiet_sample.shape}")
    print(f"  4. Quiet Sun CSV   : sample_quiet_sun_event.csv {df_quiet.shape}")
else:
    print("[ERROR] ML dataset files not found.")
