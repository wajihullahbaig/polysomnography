"""Kalman baseline, autoencoder reduction, supervised reference, t-SNE figure and the feature-space comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psng.data import SignalData
from psng.events import detect_events, flow_envelope
from psng.kalman import kalman_smooth
from psng.pipeline import Pipeline
from psng.report import _markdown_table, compare_spaces, embedding_figure
from psng.settings import EventSettings, FeatureSettings, Settings
from psng.supervised import event_classifier
from psng.transform import Autoencoder, FeatureTransformer
from tests.test_events import APNEAS, signal_with_events
from tests.test_pipeline import SFREQ


def test_kalman_level_tracks_slow_trend_and_ignores_short_dips():
    rng = np.random.default_rng(0)
    t = np.arange(3600)
    trend = 1.0 + 0.3 * np.sin(2 * np.pi * t / 3600)
    y = trend * (1 + 0.1 * rng.standard_normal(len(t)))
    y[1000:1020] *= 0.05  # a 20 s apnea
    level = kalman_smooth(y, 1e-5)
    assert np.abs(level - trend).mean() < 0.05 and level[1000:1020].min() > 0.8 * trend[1000:1020].min()
    assert kalman_smooth(np.array([])).size == 0


def test_kalman_baseline_finds_the_same_events_as_the_rolling_median():
    envelope = flow_envelope(signal_with_events())
    median = detect_events(envelope, EventSettings(baseline="rolling_median"))
    kalman = detect_events(envelope, EventSettings(baseline="kalman"))
    assert sorted(median["kind"]) == sorted(kalman["kind"]) == ["apnea", "apnea", "hypopnea"]
    for (s, _), row in zip(APNEAS, kalman[kalman["kind"] == "apnea"].sort_values("start_time").itertuples()):
        assert abs(row.start_time - s) <= 3


def test_autoencoder_reduction_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(400, 2))
    X = pd.DataFrame(np.c_[latent, latent @ rng.normal(size=(2, 4)) + 0.05 * rng.normal(size=(400, 4))], columns=list("abcdef"))
    cfg = FeatureSettings(extractor="basic", reduction={"type": "autoencoder", "n_components": 2, "max_iter": 300})
    t = FeatureTransformer(cfg).fit(X)
    Z = t.transform(X)
    assert list(Z.columns) == ["AE1", "AE2"] and np.isfinite(Z.to_numpy()).all() and Z.std().min() > 0.01
    ae = t.reducer_
    assert isinstance(ae, Autoencoder) and ae.train_error_ < 1.0
    t.save(tmp_path / "t.pkl")
    pd.testing.assert_frame_equal(FeatureTransformer.load(tmp_path / "t.pkl").transform(X), Z)


def test_supervised_reference_learns_event_windows():
    rng = np.random.default_rng(1)

    def frame(n):
        event = rng.random(n) < 0.2
        return pd.DataFrame({"window_start_time": np.arange(n) * 30.0, "window_end_time": np.arange(n) * 30.0 + 30,
                             "amplitude": np.where(event, 0.3, 1.0) + 0.1 * rng.standard_normal(n),
                             "noise": rng.standard_normal(n), "event": np.where(event, "hypopnea", "normal")})

    result, probability = event_classifier(frame(400), frame(200))
    assert probability.shape == (200,)
    assert result.validation_f1 > 0.8 and result.validation_auc > 0.9 and result.n_train == 400
    assert next(iter(result.importances)) == "amplitude"


def test_markdown_table_bolds_the_best_per_stage():
    frame = pd.DataFrame({"algorithm": ["a", "b", "a", "b"], "stage": ["training"] * 2 + ["validation"] * 2,
                          "silhouette": [0.1, 0.3, 0.4, 0.2], "davies_bouldin": [1.0, 2.0, 1.5, 0.5]})
    table = _markdown_table(frame, {"silhouette": "{:.2f}", "davies_bouldin": "{:.2f}"})
    assert "| b | training | **0.30** | 2.00 |" in table and "| b | validation | 0.20 | **0.50** |" in table


def test_compare_spaces_and_tsne(tmp_path, monkeypatch):
    signals = {"train.edf": signal_with_events(900.0).signal, "val.edf": signal_with_events(600.0).signal[::-1].copy()}
    load = lambda path, channel, start_seconds=0.0, end_seconds=None: SignalData(signal=signals[str(path)], sfreq=SFREQ, start_time=0.0, channel=channel)
    monkeypatch.setattr("psng.pipeline.load_channel", load)
    monkeypatch.setattr("psng.examples.load_channel", load)
    settings = Settings.model_validate({
        "data": {"channel": "X", "window_seconds": 30, "training": {"file_path": "train.edf"}, "validation": {"file_path": "val.edf"}},
        "features": {"extractor": "respiratory", "reduction": {"type": "pca", "n_components": 3}},
        "models": {"kmeans": {"n_clusters": 3}},
        "run": {"output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs"), "n_jobs": 1, "chunk_windows": 8, "interactive_plots": False},
    })
    for stage in ("training", "validation"):
        Pipeline(settings, stage).run()
    out = settings.run.output_dir
    assert embedding_figure(out, "kmeans", perplexity=5) == out / "comparison" / "tsne.png" and (out / "comparison" / "tsne.png").exists()

    results, markdown = compare_spaces(settings, {"pca2": {"type": "pca", "n_components": 2}, "raw": {"type": "none"},
                                                  "ae2": {"type": "autoencoder", "n_components": 2, "max_iter": 200}})
    assert set(results["space"]) == {"pca2", "raw", "ae2"} and len(results) == 6
    assert "Supervised reference" in markdown and "**" in markdown
    for name in ("spaces.csv", "spaces.png", "spaces.md", "supervised.json", "feature_importance.png", "xgboost.png"):
        assert (out / "comparison" / name).exists()
    assert (out / "spaces" / "raw" / "training" / "metrics.json").exists()
    from psng.report import latent_figure
    assert latent_figure(out, {"PCA": out / "training" / "models" / "feature_transformer.pkl",
                               "autoencoder": out / "spaces" / "ae2" / "training" / "models" / "feature_transformer.pkl"}) == out / "comparison" / "autoencoder.png"
    with pytest.raises(ValueError):
        event_classifier(pd.DataFrame({"event": ["normal"], "x": [1.0]}), pd.DataFrame({"event": ["normal"], "x": [1.0]}), target="nope")
