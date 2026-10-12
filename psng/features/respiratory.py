"""Breathing-event oriented features: airflow envelope statistics plus rhythm and shape.

Meant for windows scaled per recording (`data.normalisation: recording`), where 1.0 is the typical
breathing amplitude of the night: the envelope features then say how much airflow dropped and for
how long, which is what separates apneas / hypopneas from normal breathing. With per-window scaling
the amplitude features are nearly constant and only the rhythm / shape features carry information.
"""

from __future__ import annotations

import numpy as np

from psng.features.base import FeatureExtractor, breath_cycles, hjorth_complexity, longest_run, spectral_entropy


class RespiratoryFeatures(FeatureExtractor):
    def __init__(self, sfreq: float, hypopnea_ratio: float = 0.7, apnea_ratio: float = 0.1):
        super().__init__(sfreq)
        self.hypopnea_ratio = hypopnea_ratio
        self.apnea_ratio = apnea_ratio

    def envelope(self, window: np.ndarray) -> np.ndarray:
        """RMS amplitude per second (same units as the window)."""
        step = int(self.sfreq)
        n = len(window) // step
        if n < 1:
            return np.array([np.sqrt(np.mean(window**2))])
        return np.sqrt(np.mean(window[: n * step].reshape(n, step) ** 2, axis=1))

    def extract(self, window: np.ndarray) -> dict[str, float]:
        env = self.envelope(window)
        mean = env.mean()
        low = env < self.hypopnea_ratio
        rate, regularity = breath_cycles(window, self.sfreq)
        spectrum = np.abs(np.fft.rfft(window)) ** 2
        freqs = np.fft.rfftfreq(len(window), 1 / self.sfreq)
        return {
            "amplitude": float(np.median(env)),
            "envelope_min": float(env.min()),
            "envelope_max": float(env.max()),
            "envelope_cv": float(env.std() / mean) if mean > 0 else 0.0,
            "low_flow_fraction": float(low.mean()),
            "apnea_fraction": float((env < self.apnea_ratio).mean()),
            "longest_low_flow_seconds": float(longest_run(low)),
            "breathing_rate": rate,
            "breathing_regularity": regularity,
            "dominant_frequency": float(freqs[spectrum[1:].argmax() + 1]) if len(spectrum) > 1 else 0.0,
            "spectral_entropy": spectral_entropy(window),
            "complexity": hjorth_complexity(window),
        }
