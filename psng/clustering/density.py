"""Density-based models; unseen windows are assigned from the fitted training structure."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from psng.clustering.base import Clusterer, cluster_means, nearest_center


class DBSCANClusterer(Clusterer):
    """DBSCAN; an unseen point takes the label of the nearest core sample within eps, else noise (-1)."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        super().__init__()
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)
        self.centers_ = cluster_means(X, self.labels_)
        return self

    def predict(self, X):
        core = self.model.components_
        if len(core) == 0:
            return np.full(len(X), -1)
        distances = cdist(X, core)
        nearest = distances.argmin(axis=1)
        labels = self.model.labels_[self.model.core_sample_indices_][nearest]
        return np.where(distances[np.arange(len(X)), nearest] <= self.model.eps, labels, -1)


class KDTreeClusterer(Clusterer):
    """Radius-based density clustering over a KD-tree; unseen points go to the nearest cluster mean."""

    def __init__(self, radius: float = 0.5, min_cluster_size: int = 5, leaf_size: int = 10):
        super().__init__()
        self.radius = radius
        self.min_cluster_size = min_cluster_size
        self.leaf_size = leaf_size

    def fit(self, X):
        neighbors = KDTree(X, leafsize=self.leaf_size).query_ball_point(X, self.radius)
        labels = np.full(len(X), -1)
        cluster = 0
        for seed in range(len(X)):
            if labels[seed] != -1 or len(neighbors[seed]) < self.min_cluster_size:
                continue
            labels[seed] = cluster
            queue = [seed]
            while queue:
                point = queue.pop()
                if len(neighbors[point]) < self.min_cluster_size:
                    continue  # border point: belongs to the cluster but does not expand it
                for other in neighbors[point]:
                    if labels[other] == -1:
                        labels[other] = cluster
                        queue.append(other)
            cluster += 1
        self.labels_ = labels
        self.centers_ = cluster_means(X, labels)
        return self

    def predict(self, X):
        return nearest_center(X, self.centers_)


class IsolationForestClusterer(Clusterer):
    """DBSCAN clusters plus one extra cluster holding IsolationForest anomalies."""

    def __init__(self, contamination: float = 0.1, eps: float = 0.5, min_samples: int = 5, n_estimators: int = 100, random_state: int = 42):
        super().__init__()
        self.forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=random_state)
        self.dbscan = DBSCANClusterer(eps=eps, min_samples=min_samples)
        self.anomaly_label_ = -1

    def fit(self, X):
        anomaly = self.forest.fit_predict(X) == -1
        labels = self.dbscan.fit(X).labels_
        self.anomaly_label_ = int(labels.max()) + 1
        self.labels_ = np.where(anomaly, self.anomaly_label_, labels)
        self.centers_ = cluster_means(X, self.labels_)
        return self

    def predict(self, X):
        anomaly = self.forest.predict(X) == -1
        return np.where(anomaly, self.anomaly_label_, self.dbscan.predict(X))
