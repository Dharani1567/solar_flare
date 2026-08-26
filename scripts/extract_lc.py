"""Read a SoLEXS FITS light curve into a pandas DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_light_curve(file_path: str | Path) -> pd.DataFrame:
    """Load the first table extension of a `.lc` or `.lc.gz` FITS file.

    Column names vary by product version, so this preserves all available columns.
    """
    try:
        from astropy.io import fits
    except ImportError as error:
        raise ImportError("Install astropy to read SoLEXS FITS files: pip install astropy") from error

    file_path = Path(file_path)
    with fits.open(file_path) as hdul:
        table = hdul[1].data
        frame = pd.DataFrame({name: table[name] for name in table.names})

    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


if __name__ == "__main__":
    from utils import solexs_light_curves

    files = solexs_light_curves()
    if not files:
        print("No files found. Add .lc or .lc.gz files under data/solexs/YYYYMMDD/.")
    else:
        light_curve = read_light_curve(files[0])
        print(f"Loaded {files[0].name}: {len(light_curve)} rows")
        print(light_curve.head())
