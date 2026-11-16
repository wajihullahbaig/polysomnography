"""Statistics of discrete wavelet decomposition coefficients per level."""

from __future__ import annotations

import numpy as np
import pywt

from psng.features.base import FeatureExtractor, power_entropy


class WaveletFeatures(FeatureExtractor):
    def __init__(self, sfreq: float, wavelet: str = "db4", level: int = 2):
        super().__init__(sfreq)
        self.wavelet = wavelet
        self.level = level

    def extract(self, window: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, coeff in enumerate(pywt.wavedec(window, self.wavelet, level=self.level)):
            sq = coeff**2
            out[f"mean_level_{i}"] = float(coeff.mean())
            out[f"std_level_{i}"] = float(coeff.std())
            out[f"energy_level_{i}"] = float(sq.sum())
            out[f"entropy_level_{i}"] = power_entropy(sq)
        return out
