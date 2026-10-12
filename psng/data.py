"""Data containers shared across the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

META_COLUMNS = ("window_start_time", "window_end_time")
LABEL_COLUMNS = ("event", "stage")  # per-window reference labels added after feature extraction
EVENT_KINDS = ("normal", "hypopnea", "apnea", "artefact")
SLEEP_STAGES = ("W", "REM", "N1", "N2", "N3")


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
    # internal quality (no labels needed)
    silhouette: float | None = None
    calinski_harabasz: float | None = None
    davies_bouldin: float | None = None
    quality: str
    # agreement with rule-based airflow events (apnea / hypopnea / artefact / normal)
    event_f1: float | None = None
    event_precision: float | None = None
    event_recall: float | None = None
    event_nmi: float | None = None
    event_clusters: list[int] = []
    cluster_names: dict[str, str] = {}
    # agreement with the hypnogram, when one is configured
    stage_nmi: float | None = None
    fit_seconds: float = 0.0
    resumed: bool = False


class EventSummary(BaseModel):
    """Rule-based events of one recording segment (independent of any clusterer)."""

    stage: str
    duration_hours: float
    sleep_hours: float | None = None
    n_windows: int
    apnea: int
    hypopnea: int
    artefact: int
    events_per_hour: float  # apneas + hypopneas per recorded hour
    ahi: float | None = None  # apneas + hypopneas starting in sleep, per hour of sleep (needs a hypnogram)
    event_window_fraction: float  # windows labelled apnea or hypopnea
    baseline_amplitude: float  # median 1 Hz envelope of the segment, in signal units


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature columns of a window frame (drops window metadata and reference labels)."""
    return frame.drop(columns=list(META_COLUMNS + LABEL_COLUMNS), errors="ignore")
