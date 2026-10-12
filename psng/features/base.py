"""Feature extractor interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from scipy.signal import find_peaks


class FeatureExtractor(ABC):
    """Turns one preprocessed window into a flat dict of scalar features."""

    def __init__(self, sfreq: float):
        self.sfreq = sfreq

    @abstractmethod
    def extract(self, window: np.ndarray) -> dict[str, float]: ...


def power_entropy(power: np.ndarray) -> float:
    total = power.sum()
    return float(stats.entropy(power / total)) if total > 0 else 0.0


def spectral_entropy(window: np.ndarray) -> float:
    return power_entropy(np.abs(np.fft.rfft(window)) ** 2)


def hjorth_complexity(window: np.ndarray) -> float:
    d1 = np.diff(window)
    v0, v1, v2 = window.var(), d1.var(), np.diff(d1).var()
    return float(np.sqrt(v2 * v0) / v1) if v1 > 0 else 0.0


def breath_cycles(window: np.ndarray, sfreq: float, relative_peak_height: float = 0.25, min_breath_seconds: float = 1.5) -> tuple[float, float]:
    """Breaths per minute and coefficient of variation of breath intervals (0, 0 if fewer than 2 breaths)."""
    peaks, _ = find_peaks(
        window,
        distance=int(sfreq * min_breath_seconds),
        height=window.max() * relative_peak_height,
        prominence=0.5 * window.std(),
    )
    if len(peaks) < 2:
        return 0.0, 0.0
    intervals = np.diff(peaks) / sfreq
    return float(60.0 / intervals.mean()), float(intervals.std() / intervals.mean())


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best
