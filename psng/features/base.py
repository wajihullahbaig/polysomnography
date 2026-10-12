"""Feature extractor interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


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
