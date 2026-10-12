"""Hjorth activity, mobility and complexity."""

from __future__ import annotations

import numpy as np

from psng.features.base import FeatureExtractor


class HjorthFeatures(FeatureExtractor):
    def extract(self, window: np.ndarray) -> dict[str, float]:
        d1 = np.diff(window)
        activity, v1, v2 = window.var(), d1.var(), np.diff(d1).var()
        mobility = np.sqrt(v1 / activity) if activity > 0 else 0.0
        complexity = np.sqrt(v2 / v1) / mobility if v1 > 0 and mobility > 0 else 0.0
        return {"activity": float(activity), "mobility": float(mobility), "complexity": float(complexity)}
