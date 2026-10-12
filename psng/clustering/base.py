"""Clusterer interface with shared persistence and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.distance import cdist


class Clusterer(ABC):
    """Subclasses set `labels_` (training labels) and `centers_` (or None) in fit()."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.labels_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> Clusterer: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Clusterer:
        return joblib.load(path)


def cluster_means(X: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    ids = np.unique(labels[labels >= 0])
    return np.vstack([X[labels == k].mean(axis=0) for k in ids]) if len(ids) else None


def nearest_center(X: np.ndarray, centers: np.ndarray | None) -> np.ndarray:
    if centers is None or len(centers) == 0:
        return np.full(len(X), -1)
    return cdist(X, centers).argmin(axis=1)
