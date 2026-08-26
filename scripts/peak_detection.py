"""Detect and extract solar flare events in count-rate series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def despike_series(values: np.ndarray, kernel_size: int = 5, threshold_sigma: float = 4.0) -> np.ndarray:
    """Remove 1-second single-bin cosmic ray / particle spikes."""
    med = median_filter(values, size=kernel_size)
    diff = np.abs(values - med)
    std = np.nanstd(diff)
    cleaned = values.copy()
    spike_mask = diff > (threshold_sigma * std)
    cleaned[spike_mask] = med[spike_mask]
    return cleaned


def estimate_background_rolling(
    values: np.ndarray, window_size: int = 1800, quantile: float = 0.10
) -> np.ndarray:
    """Estimate dynamic quiet-Sun baseline using rolling quantile."""
    s = pd.Series(values)
    baseline = (
        s.rolling(window=window_size, center=True, min_periods=60)
        .quantile(quantile)
        .ffill()
        .bfill()
        .fillna(0.0)
        .to_numpy()
    )
    return baseline


def detect_flare_events(
    time_array: pd.Series | np.ndarray,
    count_array: pd.Series | np.ndarray,
    sigma_threshold: float = 4.0,
    min_duration_sec: float = 60.0,
    max_gap_sec: float = 10.0,
    savgol_window: int = 15,
) -> pd.DataFrame:
    """Detect solar flare events using dynamic background and duration constraints.

    Returns event boundaries: start_time, peak_time, end_time, duration_sec,
    peak_counts, background_counts, net_peak_counts, snr.
    """
    times = np.asarray(time_array, dtype=float)
    raw_counts = np.asarray(count_array, dtype=float)
    valid_mask = np.isfinite(raw_counts)

    if not np.any(valid_mask) or len(raw_counts) == 0:
        return pd.DataFrame(
            columns=[
                "start_time",
                "peak_time",
                "end_time",
                "duration_sec",
                "peak_counts",
                "background_counts",
                "net_peak_counts",
                "snr",
            ]
        )

    # Impute NaNs/infs for continuous filtering
    counts_filled = pd.Series(raw_counts).ffill().bfill().fillna(0.0).to_numpy()

    # 1. Despike single-bin spikes
    despiked = despike_series(counts_filled)

    # 2. Dynamic quiet-Sun background estimation
    baseline = estimate_background_rolling(despiked, window_size=1800, quantile=0.10)

    # 3. High-frequency noise smoothing
    if len(despiked) >= savgol_window:
        win_len = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        smoothed = savgol_filter(despiked, window_length=win_len, polyorder=2)
    else:
        smoothed = despiked

    # 4. Dynamic Poisson N-sigma threshold: (Signal - Baseline) > k * sqrt(Baseline)
    safe_baseline = np.maximum(baseline, 1.0)
    signal_excess = smoothed - safe_baseline
    noise_sigma = np.sqrt(safe_baseline)

    # Flare mask active only on valid data points above threshold
    flare_mask = (signal_excess > (sigma_threshold * noise_sigma)) & valid_mask

    # 5. Group contiguous points above threshold into discrete flare events
    events = []
    in_event = False
    start_idx = 0

    for i in range(len(flare_mask)):
        active = flare_mask[i]
        
        # Check for gap in time array
        gap_detected = False
        if in_event and i > 0:
            dt = times[i] - times[i - 1]
            if dt > max_gap_sec or np.isnan(dt):
                gap_detected = True

        if active and not in_event:
            in_event = True
            start_idx = i
        elif (not active or gap_detected) and in_event:
            in_event = False
            end_idx = i - 1 if not gap_detected else i - 1

            if end_idx >= start_idx:
                duration_sec = times[end_idx] - times[start_idx]
                if duration_sec >= min_duration_sec:
                    event_window = smoothed[start_idx : end_idx + 1]
                    rel_peak_idx = int(np.argmax(event_window))
                    peak_idx = start_idx + rel_peak_idx

                    events.append(
                        {
                            "start_time": float(times[start_idx]),
                            "peak_time": float(times[peak_idx]),
                            "end_time": float(times[end_idx]),
                            "duration_sec": float(duration_sec),
                            "peak_counts": float(raw_counts[peak_idx]),
                            "background_counts": float(baseline[peak_idx]),
                            "net_peak_counts": float(raw_counts[peak_idx] - baseline[peak_idx]),
                            "snr": float((smoothed[peak_idx] - baseline[peak_idx]) / noise_sigma[peak_idx]),
                        }
                    )

            if active and gap_detected:
                in_event = True
                start_idx = i

    # Handle event running until end of file
    if in_event:
        end_idx = len(flare_mask) - 1
        duration_sec = times[end_idx] - times[start_idx]
        if duration_sec >= min_duration_sec:
            event_window = smoothed[start_idx : end_idx + 1]
            rel_peak_idx = int(np.argmax(event_window))
            peak_idx = start_idx + rel_peak_idx

            events.append(
                {
                    "start_time": float(times[start_idx]),
                    "peak_time": float(times[peak_idx]),
                    "end_time": float(times[end_idx]),
                    "duration_sec": float(duration_sec),
                    "peak_counts": float(raw_counts[peak_idx]),
                    "background_counts": float(baseline[peak_idx]),
                    "net_peak_counts": float(raw_counts[peak_idx] - baseline[peak_idx]),
                    "snr": float((smoothed[peak_idx] - baseline[peak_idx]) / noise_sigma[peak_idx]),
                }
            )

    return pd.DataFrame(events)


def detect_peaks(
    counts: pd.Series | np.ndarray,
    prominence: float | None = None,
    distance: int = 1,
) -> pd.DataFrame:
    """Legacy peak detection helper maintained for backward compatibility."""
    try:
        from scipy.signal import find_peaks
    except ImportError as error:
        raise ImportError("Install scipy for peak detection: pip install scipy") from error

    values = np.asarray(counts, dtype=float)
    valid = np.isfinite(values)
    baseline = np.nanmedian(values)
    if prominence is None:
        prominence = 15

    indices, properties = find_peaks(
        np.where(valid, values, baseline), prominence=prominence, distance=distance
    )
    return pd.DataFrame(
        {
            "sample_index": indices,
            "peak_counts": values[indices],
            "prominence": properties["prominences"],
        }
    )

