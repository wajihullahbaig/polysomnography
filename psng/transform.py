"""Feature scaling, selection and reduction.

Fitted on training data only and persisted, so validation windows are projected into
the same space the clusterers were trained in (no re-fitting on unseen data).

Reduction options: `pca` (linear), `autoencoder` (non-linear bottleneck, see `Autoencoder`) or `none`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from psng.settings import FeatureSettings


class Autoencoder:
    """Symmetric MLP autoencoder (features -> hidden -> bottleneck -> hidden -> features) trained to reconstruct its input.

    Built on scikit-learn's `MLPRegressor` so no deep-learning dependency is needed; the bottleneck
    activations are the reduced features. Inputs
    should be standardised. Small data (a thousand windows) trains in a few seconds.
    """

    def __init__(self, n_components: int = 3, hidden: int = 16, activation: str = "tanh", max_iter: int = 2000, random_state: int = 42):
        self.n_components_ = n_components
        self.model = MLPRegressor(hidden_layer_sizes=(hidden, n_components, hidden), activation=activation, max_iter=max_iter,
                                  early_stopping=False, random_state=random_state, learning_rate_init=1e-3, alpha=1e-4)

    def fit(self, X: np.ndarray) -> Autoencoder:
        self.model.fit(X, X)
        self.train_error_ = float(np.mean((self.model.predict(X) - X) ** 2))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Activations of the bottleneck layer (index 1 of the hidden layers)."""
        h = np.asarray(X, dtype=float)
        for layer in range(2):  # input -> hidden -> bottleneck
            h = self._activate(h @ self.model.coefs_[layer] + self.model.intercepts_[layer])
        return h

    def _activate(self, z: np.ndarray) -> np.ndarray:
        name = self.model.activation
        if name == "relu":
            return np.maximum(z, 0)
        if name == "tanh":
            return np.tanh(z)
        if name == "logistic":
            return 1 / (1 + np.exp(-z))
        return z  # identity


class FeatureTransformer:
    def __init__(self, cfg: FeatureSettings):
        self.cfg = cfg
        self.columns_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.reducer_: PCA | Autoencoder | None = None

    @property
    def output_columns(self) -> list[str]:
        if self.reducer_ is None:
            return self.columns_
        prefix = "AE" if isinstance(self.reducer_, Autoencoder) else "PC"
        return [f"{prefix}{i + 1}" for i in range(self.reducer_.n_components_)]

    def fit(self, features: pd.DataFrame) -> FeatureTransformer:
        self.columns_ = self._select(features)
        X = features[self.columns_].to_numpy(float)
        if self.cfg.scale:
            self.scaler_ = StandardScaler().fit(X)
            X = self.scaler_.transform(X)
        red = self.cfg.reduction
        n = min(red.n_components, X.shape[1])
        if red.type == "pca":
            self.reducer_ = PCA(n_components=n).fit(X)
        elif red.type == "autoencoder":
            self.reducer_ = Autoencoder(n, red.hidden, red.activation, red.max_iter).fit(X)
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
