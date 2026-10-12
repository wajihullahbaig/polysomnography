"""Windowing and per-window preprocessing of a 1-D signal."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

RESPIRATORY_BAND_HZ = (0.1, 2.0)


def window_starts(n_samples: int, window: int, hop: int = 0) -> np.ndarray:
    """Start indices of fixed windows; hop 0 means non-overlapping."""
    if window <= 0 or window > n_samples:
        raise ValueError(f"window of {window} samples does not fit a signal of {n_samples} samples")
    return np.arange(0, n_samples - window + 1, hop or window)


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    idx, ok = np.arange(len(x)), ~np.isnan(x)
    return np.interp(idx, idx[ok], x[ok]) if ok.any() else np.zeros_like(x)


class WindowPreprocessor:
    """Smooth, detrend, band-pass (respiratory band), de-spike and scale one window.

    Scaling is either per window (z-score: every window ends up with unit variance, so only the
    *shape* of breathing survives) or per recording (`scale` = typical breathing amplitude of the
    recording, so a window during an apnea keeps its small amplitude). Recording scaling is what
    event-oriented features need. Filter designs are computed once here rather than once per window.
    """

    def __init__(self, sfreq: float, smoothing_seconds: float = 0.5, band_hz: tuple[float, float] = RESPIRATORY_BAND_HZ,
                 scale: float | None = None):
        self.sfreq = sfreq
        n = int(smoothing_seconds * sfreq) | 1  # odd kernel
        self.kernel = np.full(n, 1.0 / n) if smoothing_seconds > 0 else None
        self.sos = sps.butter(3, band_hz, btype="band", fs=sfreq, output="sos")
        self.medfilt_size = max(1, int(0.1 * sfreq)) | 1  # 100 ms
        self.scale = float(scale) if scale is not None and scale > 0 else None

    def filter(self, window: np.ndarray) -> np.ndarray:
        """Everything but the scaling, in the original units of the signal."""
        x = interpolate_nans(np.asarray(window, dtype=float))
        if self.kernel is not None:
            x = np.convolve(x, self.kernel, mode="same")
        x = sps.detrend(x)
        x = sps.sosfiltfilt(self.sos, x)
        return sps.medfilt(x, self.medfilt_size)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        x = self.filter(window)
        if self.scale is not None:
            return x / self.scale
        std = x.std()
        return x / std if std > 0 else x
