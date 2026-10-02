# Dataset Audit Report — Solar Flare Forecasting (v5)
*Generated: 2026-10-02 | Auditor: Antigravity AI*

---

## 1. Dataset Identification

| Property | Value |
|---|---|
| **Dataset Name** | ISRO Aditya-L1 Solar Flare Forecasting Dataset (v5) |
| **Source Instruments** | HEL1OS (High Energy L1 Orbiting X-ray Spectrometer) + SoLEXS (Solar Low Energy X-ray Spectrometer) aboard Aditya-L1 |
| **Data Files** | `data/ml/X_sequences_v5.npy`, `data/ml/y_labels_v5.npy`, `data/ml/sequence_metadata_v5.csv` |
| **Dataset Version** | v5 (latest, used for all training) |
| **Date Coverage** | 2024-05-14 → 2026-09-30 |
| **Observation Days** | 91 unique solar observation days |
| **Windows per Day** | 12 (2-hour windows, each 3600 seconds at 1-Hz sampling) |

---

## 2. Dataset Shape Verification

| Property | Value |
|---|---|
| **X tensor shape** | `(1092, 3600, 4)` — 1092 samples × 3600 timesteps × 4 channels |
| **y label shape** | `(1092,)` |
| **Metadata shape** | `(1092, 13)` |
| **Temporal resolution** | 1 second per timestep |
| **Window duration** | 3600 seconds (1 hour per sample) |
| **Tensor dtype** | float32 |
| **Storage size** | ~60 MB (X array) |

### Channels Description

| Channel | Instrument | Energy Band | Description |
|---|---|---|---|
| 0 | HEL1OS CdTe 1 | 10–20 keV | Hard X-ray counts (lower band) |
| 1 | HEL1OS CdTe 2 | 20–50 keV | Hard X-ray counts (mid band) |
| 2 | HEL1OS CZT 1 | 50–100 keV | Hard X-ray counts (high band) |
| 3 | HEL1OS CZT 2 / SoLEXS | 100–150 keV / 1–15 keV | Combined hard X-ray / soft X-ray |

---

## 3. Class Distribution

| Class | Label | Count | Percentage |
|---|---|---|---|
| **Quiet / Minor Event** (C-class or below) | 0 | 802 | 73.44% |
| **Major Flare** (M/X-class) | 1 | 290 | 26.56% |
| **Class Imbalance Ratio** | — | 2.77 : 1 | — |

> [!NOTE]
> The positive class (major flares) consists entirely of M-class flares (M1.0–M8.5). No X-class flares were present in the observation period. This is a moderately imbalanced dataset — class weighting is recommended but not strictly required.

---

## 4. Missing Value Audit

| Check | Result | Status |
|---|---|---|
| NaN in X tensor | 0 | ✅ PASS |
| NaN in y labels | 0 | ✅ PASS |
| Missing metadata fields | 0 (all 13 columns complete) | ✅ PASS |
| Inf values in X | 0 | ✅ PASS |
| -Inf values in X | 0 | ✅ PASS |

**Conclusion: No missing or infinite values found anywhere in the dataset.**

---

## 5. NaN & Infinite Value Scan

```
NaN count in X:     0 / 3,931,200 elements
Inf count in X:     0 / 3,931,200 elements
NaN count in y:     0 / 1,092 elements
```

All values are finite, non-NaN floating-point numbers. ✅

---

## 6. Per-Channel Statistics

| Channel | Min | Max | Mean | Std | NaN | Inf |
|---|---|---|---|---|---|---|
| 0 (CdTe 1, 10–20 keV) | 0.0080 | 7.5409 | 0.5973 | 0.8370 | 0 | 0 |
| 1 (CdTe 2, 20–50 keV) | 0.0183 | 7.5671 | 0.5972 | 0.8372 | 0 | 0 |
| 2 (CZT 1, 50–100 keV) | 0.0000 | 7.5543 | 0.5973 | 0.8369 | 0 | 0 |
| 3 (CZT 2/SoLEXS) | 0.0136 | 7.5405 | 0.5974 | 0.8371 | 0 | 0 |

> [!NOTE]
> Channel 2 has a minimum value of exactly 0.0 — this is physically plausible (zero counts in the 50–100 keV band during quiet periods) and is NOT a data corruption indicator. No channels are saturated.

---

## 7. Constant Channels

| Channel | Samples with zero intra-sequence std | Status |
|---|---|---|
| 0 | 0 / 1092 | ✅ No constant channels |
| 1 | 0 / 1092 | ✅ No constant channels |
| 2 | 0 / 1092 | ✅ No constant channels |
| 3 | 0 / 1092 | ✅ No constant channels |

**All 4 channels carry time-varying information across all 1092 samples.** ✅

---

## 8. Duplicate Sample Detection

| Metric | Value |
|---|---|
| Total samples | 1,092 |
| Unique samples (element-wise) | 1,092 |
| Duplicate samples | **0** |

**No duplicate samples found.** ✅

---

## 9. Invalid Labels

| Check | Result | Status |
|---|---|---|
| Labels outside {0, 1} | 0 | ✅ PASS |
| Label dtype | int64 | ✅ PASS |
| Unique label values | [0, 1] | ✅ PASS |

---

## 10. Flare Event ID / Grouping Verification

The metadata does **not** contain a `flare_event_id` column. However, the `date` column serves as a natural grouping key:

| Grouping Strategy | Details |
|---|---|
| **Group key** | `date` (YYYYMMDD integer format) |
| **Unique groups** | 91 observation days |
| **Samples per group** | Exactly 12 per day (fixed window scheme) |
| **Group contamination** | None — each day's 12 windows are temporally ordered and non-overlapping |

**Group-based cross-validation must use `date` as the group label** to prevent temporal leakage between windows from the same solar observation day.

### Leakage Risk Assessment
| Risk | Status |
|---|---|
| Same-day windows in train + val | ✅ Prevented via `StratifiedGroupKFold(groups=date)` |
| Data normalization using global stats | ⚠️ Must use per-fold training statistics only |
| Label derived from future data | ✅ N/A — binary label is event-level, not windowed future |

---

## 11. GOES Class Distribution Summary

- **Negative class (C-class)**: C1.0 – C4.8 and sub-C events (802 samples, 73.4%)
- **Positive class (M-class)**: M1.0 – M8.5 (290 samples, 26.6%)
- **No X-class flares** in dataset (none during observation period)
- Most frequent GOES class: C2.0 (33 samples)

---

## 12. Overall Audit Verdict

| Check | Status |
|---|---|
| Shape verified | ✅ |
| Class distribution documented | ✅ |
| NaN values | ✅ None found |
| Inf values | ✅ None found |
| Constant channels | ✅ None found |
| Duplicate samples | ✅ None found |
| Invalid labels | ✅ None found |
| Metadata completeness | ✅ 100% complete |
| Leakage grouping identified | ✅ Use `date` as group key |

> [!IMPORTANT]
> **Dataset is clean and ready for preprocessing.** No corrupted samples, no missing values, and no duplicates were found. The primary preprocessing tasks are normalization (using per-fold training statistics) and constructing proper `StratifiedGroupKFold` splits using the `date` column.

---

## 13. Recommended Preprocessing Pipeline

1. ✅ No duplicate removal needed (0 duplicates)
2. ✅ No NaN/Inf replacement needed (0 found)
3. ✅ No corrupted sample removal needed
4. **Apply per-fold z-score normalization** (fit on train fold only, transform train+val)
5. **Use `date` column as group key** for `StratifiedGroupKFold`
6. **Apply class weighting** in loss function: `pos_weight = 802/290 ≈ 2.77`

---

*Report generated by Antigravity AI agentic pipeline. Dataset: ISRO Aditya-L1 v5.*
