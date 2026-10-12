"""Feature scaling, selection and reduction.

Fitted on training data only and persisted, so validation windows are projected into
the same space the clusterers were trained in (no re-fitting on unseen data).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from psng.settings import FeatureSettings


class FeatureTransformer:
    def __init__(self, cfg: FeatureSettings):
        self.cfg = cfg
        self.columns_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.reducer_: PCA | None = None

    @property
    def output_columns(self) -> list[str]:
        return [f"PC{i + 1}" for i in range(self.reducer_.n_components_)] if self.reducer_ else self.columns_

    def fit(self, features: pd.DataFrame) -> FeatureTransformer:
        self.columns_ = self._select(features)
        X = features[self.columns_].to_numpy(float)
        if self.cfg.scale:
            self.scaler_ = StandardScaler().fit(X)
            X = self.scaler_.transform(X)
        if self.cfg.reduction.type == "pca":
            self.reducer_ = PCA(n_components=min(self.cfg.reduction.n_components, X.shape[1])).fit(X)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.columns_) - set(features.columns)
        if missing:
            raise ValueError(f"features missing columns the transformer was fitted on: {sorted(missing)}")
        X = features[self.columns_].to_numpy(float)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        if self.reducer_ is not None:
            X = self.reducer_.transform(X)
        return pd.DataFrame(X, columns=self.output_columns, index=features.index)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> FeatureTransformer:
        return joblib.load(path)

    def _select(self, features: pd.DataFrame) -> list[str]:
        sel = self.cfg.selection
        if sel.type == "correlation":
            return drop_correlated(features, sel.correlation_method, sel.correlation_threshold)
        if sel.type == "pca":
            return pca_loading_features(features, sel.pca_loading, sel.pca_variance)
        return list(features.columns)


def drop_correlated(features: pd.DataFrame, method: str, threshold: float) -> list[str]:
    """Keep a feature only if it is not correlated above `threshold` with an already kept one."""
    corr = features.corr(method=method).abs().fillna(0.0)
    kept: list[str] = []
    for column in features.columns:
        if all(corr.at[column, k] <= threshold for k in kept):
            kept.append(column)
    return kept


def pca_loading_features(features: pd.DataFrame, loading: float, variance: float) -> list[str]:
    """Features with |loading| above `loading` on any component needed to explain `variance`."""
    pca = PCA().fit(StandardScaler().fit_transform(features))
    n = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), variance)) + 1
    strong = np.abs(pca.components_[:n]).max(axis=0) > loading
    return list(features.columns[strong]) or list(features.columns)
