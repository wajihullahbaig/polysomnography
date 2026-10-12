"""Hjorth complexity, relative peak count, spectral entropy and breath-cycle statistics."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from psng.features.base import FeatureExtractor, spectral_entropy


class FreqTimePattern(FeatureExtractor):
    def __init__(self, sfreq: float, relative_peak_height: float = 0.25, min_breath_seconds: float = 1.5):
        super().__init__(sfreq)
        self.relative_peak_height = relative_peak_height
        self.min_breath_seconds = min_breath_seconds

    def extract(self, window: np.ndarray) -> dict[str, float]:
        d1 = np.diff(window)
        v0, v1, v2 = window.var(), d1.var(), np.diff(d1).var()
        rate, regularity = self._breath_cycles(window)
        return {
            "complexity": float(np.sqrt(v2 * v0) / v1) if v1 > 0 else 0.0,
            "peaks": self._relative_peak_count(window),
            "spectral_entropy": spectral_entropy(window),
            "breathing_rate": rate,
            "breathing_regularity": regularity,
        }

    def _relative_peak_count(self, window: np.ndarray) -> int:
        peaks, _ = find_peaks(window)
        if len(peaks) == 0:
            return 0
        heights = window[peaks]
        return int((heights >= heights.max() * self.relative_peak_height).sum())

    def _breath_cycles(self, window: np.ndarray) -> tuple[float, float]:
        """Breaths per minute and coefficient of variation of breath intervals."""
        peaks, _ = find_peaks(
            window,
            distance=int(self.sfreq * self.min_breath_seconds),
            height=window.max() * self.relative_peak_height,
            prominence=0.5 * window.std(),
        )
        if len(peaks) < 2:
            return 0.0, 0.0
        intervals = np.diff(peaks) / self.sfreq
        return float(60.0 / intervals.mean()), float(intervals.std() / intervals.mean())
