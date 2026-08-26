# Step-by-Step Dataset Reconstruction Guide

**Report Location**: `results/dataset_rebuild_guide.md`  

---

## Exact Commands to Rebuild `X_sequences_expanded.npy` from Scratch

```bash
# Step 1: Extract all raw Level-1 ZIP archives downloaded from ISSDC PRADAN
python scripts/extract_all_archives.py

# Step 2: Validate integrity of extracted FITS light curve files
python scripts/validate_all_fits.py

# Step 3: Match flare peak timestamps against GOES catalog and assign binary labels
python scripts/label_flares_expanded.py

# Step 4: Extract 4-channel 1 Hz light curves and construct 3,600s pre-flare sequence tensors
python scripts/build_sequence_dataset_expanded.py
```

**Expected Tensor Output**:
- `data/ml/X_sequences_expanded.npy` (Shape: `(732, 3600, 4)`)
- `data/ml/y_labels_expanded.npy` (Shape: `(732,)`, `114` Major, `618` Minor)
