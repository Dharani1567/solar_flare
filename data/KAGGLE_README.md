# ISRO Aditya-L1 Solar Flare Forecasting Dataset (v3)

## Dataset Specifications
- **Tensor Shape**: `(732, 3600, 4)` (Samples x Seconds x Channels)
- **Class Breakdown**: 114 Major M/X Flares (`y=1`), 618 Minor / Quiet Events (`y=0`)
- **Imbalance Ratio**: 5.42 : 1
- **Channels**:
  1. `CdTe 1`: High Energy Spectrometer 1 (10-150 keV)
  2. `CdTe 2`: High Energy Spectrometer 2 (10-150 keV)
  3. `CZT 1`: CZT Spectrometer Channel 1
  4. `CZT 2`: CZT Spectrometer Channel 2

## Usage in Python
```python
import numpy as np
X = np.load('X_sequences_v3.npy')
y = np.load('y_labels_v3.npy')
print("X shape:", X.shape, "y shape:", y.shape)
```
