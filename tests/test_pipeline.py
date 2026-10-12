"""Smoke tests on synthetic breathing signals; no EDF files needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_blobs

from psng.clustering import CLUSTERERS, make_clusterer
from psng.data import SignalData, feature_matrix
from psng.features import EXTRACTORS, make_extractor
from psng.metrics import evaluate
from psng.pipeline import Pipeline
from psng.settings import FeatureSettings, Settings
from psng.transform import FeatureTransformer
from psng.windows import WindowPreprocessor, window_starts

SFREQ = 256.0


def breathing(seconds: float, rate_hz: float = 0.25, apnea=(), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    x = np.sin(2 * np.pi * rate_hz * t) + 0.05 * rng.standard_normal(len(t))
    for a, b in apnea:
        x[int(a * SFREQ) : int(b * SFREQ)] *= 0.02
    return x


def make_settings(tmp_path, **models) -> Settings:
    return Settings.model_validate({
        "data": {
            "channel": "X", "window_seconds": 10, "hop_seconds": 0,
            "training": {"file_path": "train.edf", "end_seconds": 600},
            "validation": {"file_path": "val.edf", "end_seconds": 300},
        },
        "features": {"extractor": "freqtimepatt", "reduction": {"type": "pca", "n_components": 3}},
        "models": models or {"kmeans": {"n_clusters": 3}},
        "run": {"output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs"), "n_jobs": 1, "chunk_windows": 8},
    })


@pytest.fixture
def fake_edf(monkeypatch):
    signals = {
        "train.edf": breathing(600, apnea=[(100, 130), (300, 340)]),
        "val.edf": breathing(300, rate_hz=0.3, apnea=[(50, 70)], seed=1),
    }

    def load(path, channel, start_seconds=0.0, end_seconds=None):
        return SignalData(signal=signals[str(path)], sfreq=SFREQ, start_time=0.0, channel=channel)

    monkeypatch.setattr("psng.pipeline.load_channel", load)
    return signals


# ------------------------------------------------------------------ units
def test_window_starts_and_preprocessor_handle_edge_cases():
    assert window_starts(100, 10).tolist() == list(range(0, 91, 10))
    assert window_starts(100, 10, 5).tolist() == list(range(0, 91, 5))
    with pytest.raises(ValueError):
        window_starts(5, 10)
    prep = WindowPreprocessor(SFREQ)
    flat = prep(np.zeros(int(10 * SFREQ)))
    assert np.isfinite(flat).all()
    with_nan = breathing(10)
    with_nan[100:120] = np.nan
    assert np.isfinite(prep(with_nan)).all()


@pytest.mark.parametrize("name", sorted(EXTRACTORS))
def test_extractors_return_finite_features(name):
    extractor = make_extractor(name, SFREQ)
    prep = WindowPreprocessor(SFREQ)
    for window in (breathing(10), np.zeros(int(10 * SFREQ)), np.random.default_rng(1).standard_normal(int(10 * SFREQ))):
        feats = extractor.extract(prep(window))
        assert feats and all(np.isfinite(v) for v in feats.values()), feats


def test_breathing_rate_is_per_minute():
    feats = make_extractor("freqtimepatt", SFREQ).extract(WindowPreprocessor(SFREQ)(breathing(30, rate_hz=0.25)))
    assert feats["breathing_rate"] == pytest.approx(15.0, rel=0.1)


def test_transformer_is_fitted_once_and_reused(tmp_path):
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    val = pd.DataFrame(rng.normal(loc=3.0, size=(50, 5)), columns=list("abcde"))
    cfg = FeatureSettings(extractor="basic", reduction={"type": "pca", "n_components": 2})
    t = FeatureTransformer(cfg).fit(train)
    np.testing.assert_allclose(t.scaler_.mean_, train.mean().to_numpy())  # scaler from training only
    t.save(tmp_path / "t.pkl")
    loaded = FeatureTransformer.load(tmp_path / "t.pkl")
    pd.testing.assert_frame_equal(t.transform(val), loaded.transform(val))
    assert list(t.transform(val).columns) == ["PC1", "PC2"]
    assert t.transform(val)["PC1"].abs().mean() > 1.0  # shifted data is NOT re-centred on itself
    with pytest.raises(ValueError):
        t.transform(val.drop(columns="a"))


def test_selection_strategies():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "a_copy": a * 2 + 1e-9, "b": rng.normal(size=300)})
    corr = FeatureTransformer(FeatureSettings(extractor="basic", selection={"type": "correlation", "correlation_threshold": 0.9})).fit(df)
    assert corr.columns_ == ["a", "b"]
    pca = FeatureTransformer(FeatureSettings(extractor="basic", selection={"type": "pca"})).fit(df)
    assert set(pca.columns_) <= set(df.columns) and pca.columns_


@pytest.mark.parametrize("name", sorted(CLUSTERERS))
def test_clusterers_fit_predict_and_roundtrip(name, tmp_path):
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.3, random_state=0)
    params = {"n_components": 3} if name == "gmm" else {"n_clusters": 3} if name in ("kmeans", "minibatch", "bisecting") else {}
    model = make_clusterer(name, params).fit(X)
    assert model.params == params
    assert len(model.labels_) == len(X)
    new = model.predict(X[:10] + 0.01)
    assert new.shape == (10,)
    model.save(tmp_path / "m.pkl")
    np.testing.assert_array_equal(model.load(tmp_path / "m.pkl").predict(X[:10]), model.predict(X[:10]))
    assert evaluate(X, model.labels_, algorithm=name, stage="training").n_clusters >= 2


def test_metrics_ignore_noise_label():
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.3, random_state=0)
    y[:10] = -1
    m = evaluate(X, y, algorithm="x", stage="training")
    assert m.n_noise == 10 and m.n_clusters == 2 and m.silhouette is not None and m.quality in {"good", "excellent"}


def test_settings_reject_unknown_names(tmp_path):
    with pytest.raises(ValidationError):
        make_settings(tmp_path, nonsense={})
    with pytest.raises(ValidationError):
        Settings.model_validate({**make_settings(tmp_path).model_dump(), "features": {"extractor": "nope"}})


# ------------------------------------------------------------------ pipeline
def test_training_validation_and_resume(tmp_path, fake_edf):
    settings = make_settings(tmp_path, kmeans={"n_clusters": 3}, dbscan={"eps": 1.0, "min_samples": 3})
    train = Pipeline(settings, "training").run()
    assert {m.algorithm for m in train} == {"kmeans", "dbscan"} and not any(m.resumed for m in train)
    out = settings.run.output_dir
    assert (out / "training" / "models" / "feature_transformer.pkl").exists()
    for name in ("kmeans", "dbscan"):
        assert (out / "training" / "csvs" / f"{name}.csv").exists()
        assert (out / "training" / "plots" / f"{name}.png").exists()
        assert (out / "training" / "plots" / f"{name}.html").exists()
    assert (out / "training" / "metrics.json").exists()

    model_mtime = (out / "training" / "models" / "kmeans.pkl").stat().st_mtime_ns
    resumed = Pipeline(settings, "training").run()
    assert all(m.resumed for m in resumed)
    assert (out / "training" / "models" / "kmeans.pkl").stat().st_mtime_ns == model_mtime

    val = Pipeline(settings, "validation").run()
    labels = pd.read_csv(out / "validation" / "csvs" / "kmeans.csv")
    assert len(labels) == 30 and labels["cluster_id"].between(0, 2).all()
    assert {m.algorithm for m in val} == {"kmeans", "dbscan"}

    settings.run.resume = False  # --no-resume refits
    assert not any(m.resumed for m in Pipeline(settings, "training").run())


def test_validation_without_training_fails_clearly(tmp_path, fake_edf):
    with pytest.raises(FileNotFoundError):
        Pipeline(make_settings(tmp_path), "validation").run()


def test_feature_cache_resumes_mid_extraction(tmp_path, fake_edf):
    settings = make_settings(tmp_path)
    pipeline = Pipeline(settings, "training")
    signal = SignalData(signal=fake_edf["train.edf"], sfreq=SFREQ, start_time=0.0, channel="X")
    full = pipeline.extract_features(signal)
    assert len(full) == 60

    cache = pipeline.cache_path()
    partial = pd.read_csv(cache).iloc[:13]
    partial.to_csv(cache, index=False)
    with open(cache, "ab") as f:
        f.write(b"0.0,garbage")  # interrupted write without newline
    resumed = Pipeline(settings, "training").extract_features(signal)
    pd.testing.assert_frame_equal(resumed, full, check_dtype=False)  # int column re-read as float after rewrite
    assert feature_matrix(resumed).shape[1] == 5
