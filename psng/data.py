"""Data containers shared across the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

META_COLUMNS = ("window_start_time", "window_end_time")


class SignalData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    signal: np.ndarray
    sfreq: float
    start_time: float
    channel: str

    @property
    def duration(self) -> float:
        return len(self.signal) / self.sfreq


class ClusterMetrics(BaseModel):
    algorithm: str
    stage: str
    n_windows: int
    n_clusters: int
    n_noise: int = 0
    silhouette: float | None = None
    calinski_harabasz: float | None = None
    davies_bouldin: float | None = None
    quality: str
    fit_seconds: float = 0.0
    resumed: bool = False


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature columns of a window frame (drops window metadata)."""
    return frame.drop(columns=list(META_COLUMNS), errors="ignore")
