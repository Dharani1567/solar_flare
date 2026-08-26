# HEL1OS 1 Hz Raw Time-Series Sequence Dataset Verification Report

**Dataset Generation Target**: Raw Deep Learning Input Tensors  
**Output Path (X)**: [X_sequences.npy](file:///home/dharani/Desktop/solar_flare/data/ml/X_sequences.npy)  
**Output Path (y)**: [y_labels.npy](file:///home/dharani/Desktop/solar_flare/data/ml/y_labels.npy)  
**Metadata Table**: [sequence_metadata.csv](file:///home/dharani/Desktop/solar_flare/data/ml/sequence_metadata.csv)  

---

## 1. Tensor Specifications & Verification Checklist

| Metric / Check | Value / Status | Verification Result |
| :--- | :--- | :--- |
| **Number of Sequence Samples ($N$)** | **474** | Flares with complete 60-minute HEL1OS coverage |
| **Sequence Length ($T$)** | **3600** | Exactly **3,600 seconds (60 minutes)** at 1 Hz cadence |
| **Input Channels ($C$)** | **4** | `cdte1`, `cdte2`, `czt1`, `czt2` |
| **X Tensor Shape** | **`(474, 3600, 4)`** | `(N, 3600, 4)` |
| **y Label Tensor Shape** | **`(474,)`** | `(N,)` |
| **NaN / Missing Values Count** | **0** | **VERIFIED CLEAN (0 NaNs)** |
| **Infinite Values Count** | **0** | **VERIFIED CLEAN (0 Inf)** |

---

## 2. Target Class Balance (`binary_major_flare`)

| Class | Description | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0 (Minor Flares)** | B Class, C Class, & UNMATCHED Flares | **396** | **83.54%** |
| **1 (Major Flares)** | M Class & X Class Flares | **78** | **16.46%** |
| **Total** | Validated Sequence Flares | **474** | **100.00%** |

*Class Imbalance Ratio*: **5.08 : 1** (16.46% positive class).

---

## 3. Raw Signal Distribution & Channel Breakdown

- **Global Min Count Rate**: `0.0000` counts / sec
- **Global Max Count Rate**: `9038.7998` counts / sec
- **Global Mean Count Rate**: `23.9013` counts / sec
- **Global Std Count Rate**: `135.5614` counts / sec

### Channel Index Mapping:
- Index `0`: `cdte1` (Soft/Hard X-ray, 1.8 – 90.0 keV)
- Index `1`: `cdte2` (Soft/Hard X-ray, 1.8 – 90.0 keV)
- Index `2`: `czt1` (Hard X-ray, 18.0 – 160.0 keV)
- Index `3`: `czt2` (Hard X-ray, 18.0 – 160.0 keV)

---

## 4. Deep Learning Readiness Verification

- [x] **Raw Time Series Intact**: 3,600 continuous 1 Hz count rate samples per sequence. Zero pre-aggregation loss.
- [x] **Multi-Channel Alignment**: Synchronized timestamps across CdTe1, CdTe2, CZT1, CZT2 detectors.
- [x] **No Temporal Data Leakage**: Sequences span strictly $[t_{\text{peak}} - 3600\text{s}, t_{\text{peak}}]$. Zero post-peak data.
- [x] **Ready for PyTorch 1D-CNN / LSTM / Transformer Training**: Tensor files can be loaded via `np.load()` directly into PyTorch `Dataset` / `DataLoader`.
