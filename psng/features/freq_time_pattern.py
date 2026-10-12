"""Hjorth complexity, relative peak count, spectral entropy and breath-cycle statistics."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from psng.features.base import FeatureExtractor, breath_cycles, hjorth_complexity, spectral_entropy


class FreqTimePattern(FeatureExtractor):
    def __init__(self, sfreq: float, relative_peak_height: float = 0.25, min_breath_seconds: float = 1.5):
        super().__init__(sfreq)
        self.relative_peak_height = relative_peak_height
        self.min_breath_seconds = min_breath_seconds

    def extract(self, window: np.ndarray) -> dict[str, float]:
        rate, regularity = breath_cycles(window, self.sfreq, self.relative_peak_height, self.min_breath_seconds)
        return {
            "complexity": hjorth_complexity(window),
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
