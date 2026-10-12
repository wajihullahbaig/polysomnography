"""Clusterer registry: settings.yaml `models` keys map here; values are constructor arguments."""

from typing import Any

from psng.clustering.base import Clusterer
from psng.clustering.density import DBSCANClusterer, IsolationForestClusterer, KDTreeClusterer
from psng.clustering.sklearn_models import BisectingKMeansClusterer, GMMClusterer, KMeansClusterer, MiniBatchKMeansClusterer

CLUSTERERS: dict[str, type[Clusterer]] = {
    "kmeans": KMeansClusterer,
    "minibatch": MiniBatchKMeansClusterer,
    "bisecting": BisectingKMeansClusterer,
    "gmm": GMMClusterer,
    "dbscan": DBSCANClusterer,
    "kdtree": KDTreeClusterer,
    "isolation": IsolationForestClusterer,
}


def make_clusterer(name: str, params: dict[str, Any]) -> Clusterer:
    clusterer = CLUSTERERS[name](**params)
    clusterer.params = dict(params)
    return clusterer
