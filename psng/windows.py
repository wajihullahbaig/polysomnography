"""Windowing and per-window preprocessing of a 1-D signal."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def window_starts(n_samples: int, window: int, hop: int = 0) -> np.ndarray:
    """Start indices of fixed windows; hop 0 means non-overlapping."""
    if window <= 0 or window > n_samples:
        raise ValueError(f"window of {window} samples does not fit a signal of {n_samples} samples")
    return np.arange(0, n_samples - window + 1, hop or window)


class WindowPreprocessor:
    """Smooth, detrend, band-pass (respiratory band), de-spike and z-scale one window.

    PSG signals are non-stationary, so each window is normalised on its own.
    Filter designs are computed once here rather than once per window.
    """

    def __init__(self, sfreq: float, smoothing_seconds: float = 0.5, band_hz: tuple[float, float] = (0.1, 2.0)):
        self.sfreq = sfreq
        n = int(smoothing_seconds * sfreq) | 1  # odd kernel
        self.kernel = np.full(n, 1.0 / n) if smoothing_seconds > 0 else None
        self.sos = sps.butter(3, band_hz, btype="band", fs=sfreq, output="sos")
        self.medfilt_size = max(1, int(0.1 * sfreq)) | 1  # 100 ms

    def __call__(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=float)
        if np.isnan(x).any():
            idx, ok = np.arange(len(x)), ~np.isnan(x)
            x = np.interp(idx, idx[ok], x[ok]) if ok.any() else np.zeros_like(x)
        if self.kernel is not None:
            x = np.convolve(x, self.kernel, mode="same")
        x = sps.detrend(x)
        x = sps.sosfiltfilt(self.sos, x)
        x = sps.medfilt(x, self.medfilt_size)
        std = x.std()
        return x / std if std > 0 else x
