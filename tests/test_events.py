"""Event detection, hypnogram handling, reference scores, cluster names and the report on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psng.data import SignalData, feature_matrix
from psng.events import (cluster_profile, detect_events, event_detection_scores, event_summary, flow_envelope, load_hypnogram,
                         name_clusters, recording_baseline, reference_scores, window_event_labels, window_stages)
from psng.features import make_extractor
from psng.pipeline import Pipeline
from psng.report import build_report, sweep
from psng.settings import EventSettings, Settings
from psng.windows import WindowPreprocessor
from tests.test_pipeline import SFREQ, breathing

APNEAS = [(100.0, 125.0), (300.0, 340.0)]  # (start, end) seconds; amplitude x0.02
HYPOPNEA = (450.0, 470.0)  # amplitude x0.5


def signal_with_events(seconds: float = 600.0) -> SignalData:
    x = breathing(seconds, apnea=APNEAS)
    x[int(HYPOPNEA[0] * SFREQ) : int(HYPOPNEA[1] * SFREQ)] *= 0.5
    return SignalData(signal=x, sfreq=SFREQ, start_time=0.0, channel="X")


def hypnogram_file(tmp_path, n_epochs: int = 20, stages=("W",) * 5 + ("N1",) * 3 + ("N2",) * 8 + ("REM",) * 4):
    path = tmp_path / "hypno.txt"
    path.write_text("".join(f"{30 * (i + 1)}\t00:00:00\t{s}  \t1\n" for i, s in enumerate(stages[:n_epochs])), encoding="utf-8")
    return path


def test_detector_finds_apneas_and_hypopnea_with_correct_timing():
    signal = signal_with_events()
    envelope = flow_envelope(signal)
    assert len(envelope) == 600 and recording_baseline(envelope) > 0
    events = detect_events(envelope, EventSettings())
    found = {row.kind: (row.start_time, row.end_time) for row in events.itertuples()}
    apneas = events[events["kind"] == "apnea"].sort_values("start_time")
    assert len(apneas) == 2 and (events["kind"] == "hypopnea").sum() == 1 and not (events["kind"] == "artefact").any()
    for (s, e), row in zip(APNEAS, apneas.itertuples()):
        assert abs(row.start_time - s) <= 3 and abs(row.end_time - e) <= 3 and row.min_flow < 0.1
    hs, he = found["hypopnea"]
    assert abs(hs - HYPOPNEA[0]) <= 3 and abs(he - HYPOPNEA[1]) <= 3


def test_long_low_flow_is_an_artefact_and_short_dips_are_ignored():
    x = breathing(600)
    x[int(200 * SFREQ) : int(400 * SFREQ)] *= 0.01  # 200 s flat
    x[int(50 * SFREQ) : int(55 * SFREQ)] *= 0.01  # 5 s dip, below min_seconds
    events = detect_events(flow_envelope(SignalData(signal=x, sfreq=SFREQ, start_time=0.0, channel="X")), EventSettings())
    assert list(events["kind"]) == ["artefact"]


def test_window_labels_and_stages(tmp_path):
    starts = np.arange(0, 600, 30.0)
    ends = starts + 30
    events = pd.DataFrame({"start_time": [100.0, 450.0], "end_time": [125.0, 470.0], "duration": [25.0, 20.0], "kind": ["apnea", "hypopnea"], "min_flow": [0.0, 0.5]})
    labels = window_event_labels(starts, ends, events, 5.0)
    assert labels[3] == "apnea" and labels[4] == "apnea" and labels[15] == "hypopnea" and labels[0] == "normal"
    assert window_event_labels(starts, ends, events, 25.0)[4] == "normal"  # only 5 s of overlap in window 4

    hypnogram = load_hypnogram(hypnogram_file(tmp_path))
    assert list(hypnogram.columns) == ["end_time", "stage"] and hypnogram["stage"].iloc[0] == "W" and hypnogram["stage"].iloc[-1] == "REM"
    stages = window_stages(starts, ends, hypnogram)
    assert stages[0] == "W" and stages[5] == "N1" and stages[19] == "REM"
    assert window_stages(np.array([1000.0]), np.array([1030.0]), hypnogram)[0] == "?"

    summary = event_summary("training", events, labels, stages, 600.0, 1.0, hypnogram)
    assert summary.apnea == 1 and summary.hypopnea == 1 and summary.ahi is not None
    assert summary.sleep_hours == pytest.approx(15 * 30 / 3600)
    assert summary.ahi == pytest.approx(1 / summary.sleep_hours)  # the apnea is in wake, the hypopnea in N2


def test_reference_and_detection_scores():
    events = np.array(["normal"] * 50 + ["apnea"] * 10 + ["hypopnea"] * 10)
    perfect = np.array([0] * 50 + [1] * 20)
    assert event_detection_scores(perfect, events)["event_f1"] == 1.0 and event_detection_scores(perfect, events)["event_clusters"] == [1]
    assert reference_scores(perfect, events)["purity"] == pytest.approx(60 / 70)  # apnea and hypopnea share cluster 1
    mixed = np.array([0] * 60 + [1] * 10)  # cluster 0 swallows the apneas
    scores = event_detection_scores(mixed, events)
    assert scores["event_recall"] == 0.5 and scores["event_precision"] == 1.0
    noisy = np.array([-1] * 70)
    assert reference_scores(noisy, events)["nmi"] is None


def test_profile_and_cluster_names():
    labels = np.array([0] * 30 + [1] * 20 + [2] * 10)
    events = np.array(["normal"] * 30 + ["apnea"] * 15 + ["hypopnea"] * 5 + ["normal"] * 10)
    stages = np.array(["N2"] * 30 + ["REM"] * 20 + ["W"] * 10)
    features = pd.DataFrame({"amplitude": [1.0] * 30 + [0.1] * 20 + [1.2] * 10, "breathing_rate": [15.0] * 50 + [30.0] * 10,
                             "breathing_regularity": [0.1] * 60})
    profile = cluster_profile(labels, events, stages, features)
    assert list(profile["cluster_id"]) == [0, 1, 2] and profile.loc[1, "event_rate"] == 1.0 and profile.loc[0, "n_N2"] == 30
    names = name_clusters(profile)
    assert names == {0: "regular breathing, mostly N2", 1: "apnea, mostly REM", 2: "fast breathing, mostly W"}
    duplicate = name_clusters(pd.concat([profile.iloc[[0]], profile.iloc[[0]].assign(cluster_id=5)], ignore_index=True))
    assert duplicate[5].endswith("#2")


def test_respiratory_features_see_amplitude_with_recording_scaling():
    extractor = make_extractor("respiratory", SFREQ)
    normal = breathing(30)
    apnea = breathing(30, apnea=[(5, 25)])
    scaled = WindowPreprocessor(SFREQ, scale=0.7)  # roughly the RMS of a unit sine
    f_normal, f_apnea = extractor.extract(scaled(normal)), extractor.extract(scaled(apnea))
    assert f_normal["amplitude"] == pytest.approx(1.0, abs=0.2) and f_normal["low_flow_fraction"] < 0.1
    assert f_apnea["longest_low_flow_seconds"] >= 15 and f_apnea["apnea_fraction"] > 0.4
    assert f_normal["breathing_rate"] == pytest.approx(15.0, rel=0.1)
    per_window = WindowPreprocessor(SFREQ)  # z-scored: amplitude is uninformative but finite
    assert all(np.isfinite(v) for v in extractor.extract(per_window(apnea)).values())


def test_pipeline_with_events_hypnogram_names_report_and_sweep(tmp_path, monkeypatch):
    signals = {"train.edf": signal_with_events(600.0).signal, "val.edf": signal_with_events(600.0).signal[::-1].copy()}
    monkeypatch.setattr("psng.pipeline.load_channel", lambda path, channel, start_seconds=0.0, end_seconds=None:
                        SignalData(signal=signals[str(path)], sfreq=SFREQ, start_time=0.0, channel=channel))
    monkeypatch.setattr("psng.examples.load_channel", lambda path, channel, start_seconds=0.0, end_seconds=None:
                        SignalData(signal=signals[str(path)], sfreq=SFREQ, start_time=0.0, channel=channel))
    settings = Settings.model_validate({
        "data": {"channel": "X", "window_seconds": 30, "normalisation": "recording",
                 "training": {"file_path": "train.edf", "hypnogram_path": str(hypnogram_file(tmp_path))},
                 "validation": {"file_path": "val.edf"}},
        "features": {"extractor": "respiratory", "reduction": {"type": "pca", "n_components": 3}},
        "models": {"kmeans": {"n_clusters": 3}, "dbscan": {"eps": 1.0, "min_samples": 3}},
        "run": {"output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs"), "n_jobs": 1, "chunk_windows": 8, "interactive_plots": True},
    })
    train = Pipeline(settings, "training").run()
    out = settings.run.output_dir
    assert (out / "training" / "events.csv").exists() and (out / "training" / "events.json").exists() and (out / "training" / "plots" / "events.png").exists()
    kmeans = next(m for m in train if m.algorithm == "kmeans")
    assert kmeans.event_f1 is not None and kmeans.stage_nmi is not None and set(kmeans.cluster_names) == {"0", "1", "2"}
    frame = pd.read_csv(out / "training" / "csvs" / "kmeans.csv")
    assert {"event", "stage", "cluster_label"} <= set(frame.columns) and "apnea" in set(frame["event"])
    assert not frame["cluster_label"].str.startswith("cluster_label").any()
    assert feature_matrix(frame.drop(columns=["cluster_id", "cluster_label"])).shape[1] == 12
    profile = pd.read_csv(out / "training" / "csvs" / "kmeans_clusters.csv")
    assert {"name", "event_rate", "n_apnea", "n_W", "mean_amplitude"} <= set(profile.columns)
    assert (out / "training" / "plots" / "kmeans_timeline.png").exists()

    val = Pipeline(settings, "validation").run()
    assert next(m for m in val if m.algorithm == "kmeans").cluster_names == kmeans.cluster_names  # names come from training
    assert next(m for m in val if m.algorithm == "kmeans").stage_nmi is None  # no hypnogram for validation

    markdown = build_report(out, tmp_path / "docs", settings)
    assert "| algorithm |" in markdown and "training" in markdown and "validation" in markdown
    assert "Kalman baseline" in markdown and (out / "comparison" / "baselines.csv").exists() and (out / "comparison" / "kalman.png").exists()
    for name in ("metrics.png", "event_rates.png", "event_examples.png", "kmeans_examples.png", "dbscan_examples.png", "report.md"):
        assert (out / "comparison" / name).exists()
    assert (tmp_path / "docs" / "figures" / "training_kmeans_timeline.png").exists()

    results = sweep(settings, "kmeans", [2, 3, 4])
    assert list(results["k"]) == [2, 3, 4] and results["silhouette"].notna().all()
    assert (out / "training" / "sweep_kmeans.png").exists()
    with pytest.raises(ValueError):
        sweep(settings, "dbscan", [2])
