"""Centroid / mixture models that natively predict unseen data."""

from __future__ import annotations

from sklearn.cluster import BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from psng.clustering.base import Clusterer


class SklearnClusterer(Clusterer):
    """Wraps an sklearn estimator exposing fit/predict and cluster_centers_ or means_."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.centers_ = getattr(self.model, "cluster_centers_", getattr(self.model, "means_", None))
        return self

    def predict(self, X):
        return self.model.predict(X)


class KMeansClusterer(SklearnClusterer):
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        super().__init__(KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state))


class MiniBatchKMeansClusterer(SklearnClusterer):
    def __init__(self, n_clusters: int = 8, batch_size: int = 256, max_iter: int = 100, random_state: int = 42):
        super().__init__(MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter, n_init="auto", random_state=random_state))


class BisectingKMeansClusterer(SklearnClusterer):
    def __init__(self, n_clusters: int = 8, bisecting_strategy: str = "biggest_inertia", n_init: int = 10, random_state: int = 42):
        super().__init__(BisectingKMeans(n_clusters=n_clusters, bisecting_strategy=bisecting_strategy, n_init=n_init, random_state=random_state))


class GMMClusterer(SklearnClusterer):
    def __init__(self, n_components: int = 8, covariance_type: str = "full", max_iter: int = 100, reg_covar: float = 1e-6, random_state: int = 42):
        super().__init__(GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter, reg_covar=reg_covar, random_state=random_state))
