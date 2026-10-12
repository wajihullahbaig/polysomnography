"""Clustering quality metrics: internal (silhouette family) and against reference labels (events, sleep stages)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from psng.data import ClusterMetrics
from psng.events import event_detection_scores, reference_scores

TABLE_COLUMNS = ["algorithm", "stage", "n_windows", "n_clusters", "n_noise", "silhouette", "calinski_harabasz", "davies_bouldin",
                 "quality", "event_f1", "event_nmi", "stage_nmi", "fit_seconds", "resumed"]


def evaluate(X: np.ndarray, labels: np.ndarray, *, algorithm: str, stage: str, fit_seconds: float = 0.0, resumed: bool = False,
             event_labels: np.ndarray | None = None, stages: np.ndarray | None = None) -> ClusterMetrics:
    """Noise windows (label -1) are excluded from every score."""
    labels = np.asarray(labels)
    mask = labels >= 0
    n_clusters = len(np.unique(labels[mask]))
    metrics = ClusterMetrics(
        algorithm=algorithm, stage=stage, n_windows=len(labels), n_clusters=n_clusters,
        n_noise=int((~mask).sum()), quality="invalid (single cluster)", fit_seconds=fit_seconds, resumed=resumed,
    )
    if n_clusters > 1 and mask.sum() > n_clusters:
        Xm, lm = X[mask], labels[mask]
        metrics.silhouette = float(silhouette_score(Xm, lm, sample_size=min(len(lm), 5000), random_state=0))
        metrics.calinski_harabasz = float(calinski_harabasz_score(Xm, lm))
        metrics.davies_bouldin = float(davies_bouldin_score(Xm, lm))
        metrics.quality = describe(metrics.silhouette)
    if event_labels is not None and n_clusters >= 1:
        detection = event_detection_scores(labels, event_labels)
        metrics.event_f1, metrics.event_precision, metrics.event_recall = detection["event_f1"], detection["event_precision"], detection["event_recall"]
        metrics.event_clusters = detection["event_clusters"]
        metrics.event_nmi = reference_scores(labels, event_labels)["nmi"]
    if stages is not None and n_clusters >= 1:
        metrics.stage_nmi = reference_scores(labels, stages)["nmi"]
    return metrics


def describe(silhouette: float) -> str:
    for threshold, label in ((0.7, "excellent"), (0.5, "good"), (0.25, "fair"), (0.0, "poor")):
        if silhouette > threshold:
            return label
    return "bad"


def write_metrics(metrics: list[ClusterMetrics], path: Path) -> None:
    path.write_text(json.dumps([m.model_dump() for m in metrics], indent=2), encoding="utf-8")


def read_metrics(path: Path) -> list[ClusterMetrics]:
    return [ClusterMetrics.model_validate(row) for row in json.loads(path.read_text(encoding="utf-8"))]


def format_table(metrics: list[ClusterMetrics]) -> str:
    frame = pd.DataFrame([m.model_dump() for m in metrics])
    columns = [c for c in TABLE_COLUMNS if c in frame.columns]
    return frame[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}")
