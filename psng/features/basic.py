"""Amplitude and peak statistics."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from psng.features.base import FeatureExtractor


class BasicFeatures(FeatureExtractor):
    def __init__(self, sfreq: float, min_prominence: float = 0.1):
        super().__init__(sfreq)
        self.min_prominence = min_prominence

    def extract(self, window: np.ndarray) -> dict[str, float]:
        peaks, _ = find_peaks(window, prominence=self.min_prominence)
        valleys, _ = find_peaks(-window, prominence=self.min_prominence)
        rms = float(np.sqrt(np.mean(window**2)))
        peak_amplitude = float(np.abs(window).max())
        crest_factor = peak_amplitude / rms if rms > 0 else 0.0
        alpha = 1.0 / (1.0 + np.exp(-(crest_factor - 2.0)))  # weight peaks more when the signal is spiky
        return {
            "mean": float(window.mean()),
            "variance": float(window.var()),
            "peak_count": len(peaks),
            "valley_count": len(valleys),
            "peak_amplitude": peak_amplitude,
            "crest_factor": crest_factor,
            "sustained_power_ratio": float((np.abs(window) > rms).mean()),
            "integrated_power": float(alpha * peak_amplitude + (1 - alpha) * rms),
        }
