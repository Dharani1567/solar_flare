# Complete Data Lineage Audit Report

**Report Location**: `results/data_lineage_report.md`  

---

## 1. Data Lineage Flowchart

```
Level-1 Raw Archives (.zip from ISSDC PRADAN)
                   │
                   ▼ (scripts/extract_all_archives.py)
Extracted FITS Light Curves (data/hel1os/extracted/YYYYMMDD/*.fits)
                   │
                   ▼ (scripts/validate_all_fits.py)
Validated Uncorrupted FITS Light Curves (4,491 verified FITS)
                   │
                   ▼ (scripts/build_sequence_dataset_expanded.py)
60-Minute Windowing & GOES Class Matching (Lookback: 3600s, N >= 2500 samples)
                   │
                   ▼
Final Machine Learning Tensors (X_sequences_expanded.npy: (732, 3600, 4), y_labels_expanded.npy)
```

---

## 2. Processing Pipeline Transformation Table

| Pipeline Stage | Input Artifacts | Script Executed | Output Artifacts | Quality Validation Check |
| :--- | :--- | :--- | :--- | :--- |
| **1. Extraction** | Raw `.zip` archives in `data/hel1os/raw/123/` | `scripts/extract_all_archives.py` | Extracted date folders in `data/hel1os/extracted/` | ZIP checksum verification |
| **2. FITS Validation** | Raw `.fits` files | `scripts/validate_all_fits.py` | `results/fits_validation_report.md` | Astropy header parse & nan check |
| **3. Flare Labeling** | `data/solexs/flare_catalog/` & GOES CSVs | `scripts/label_flares_expanded.py` | `flare_events_labeled_expanded.csv` | GOES class matching ($M, X ightarrow 1$) |
| **4. Sequence Building** | Validated FITS + Labeled Catalog | `scripts/build_sequence_dataset_expanded.py` | `X_sequences_expanded.npy`, `y_labels_expanded.npy` | Lookback sample count $N \ge 2,500$ |
