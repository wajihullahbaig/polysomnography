"""SpO2 desaturations, event confirmation and detector scores on synthetic oximetry."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from psng.data import SignalData
from psng.oximetry import (confirmed, desaturation_scores, detect_desaturations, event_desaturation, explained, oximetry_summary, spo2_series,
                           window_intervals)
from psng.pipeline import Pipeline
from psng.report import build_report
from psng.settings import OximetrySettings, Settings
from tests.test_events import APNEAS, HYPOPNEA, hypnogram_file, signal_with_events
from tests.test_pipeline import SFREQ

CFG = OximetrySettings()
NADIRS = [e + 20 for _, e in APNEAS] + [HYPOPNEA[1] + 20]  # each event desaturates 20 s after it ends


def spo2_signal(seconds: float = 600.0, nadirs=NADIRS, depth: float = 5.0, dropout=None) -> SignalData:
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    x = np.full(len(t), 95.0)
    for n in nadirs:
        x -= depth * np.clip(1 - np.abs(t - n) / 10, 0, None)  # 20 s V-shaped dip
    x = np.round(x * 2) / 2  # sensor quantisation
    if dropout is not None:
        x[int(dropout[0] * SFREQ) : int(dropout[1] * SFREQ)] = 0.5  # probe off
    return SignalData(signal=x, sfreq=SFREQ, start_time=0.0, channel="SAT")


def test_spo2_series_drops_probe_off_and_bridges_short_gaps():
    short = spo2_series(spo2_signal(dropout=(200, 205)), CFG)
    assert len(short) == 600 and short["spo2"].notna().all() and short["spo2"].between(89, 96).all()
    long = spo2_series(spo2_signal(dropout=(200, 260)), CFG)
    assert long["spo2"].isna().sum() >= 40


def test_desaturations_found_at_the_nadirs_and_flicker_ignored():
    desaturations = detect_desaturations(spo2_series(spo2_signal(), CFG), CFG)
    assert len(desaturations) == len(NADIRS)
    assert np.allclose(desaturations["nadir_time"], NADIRS, atol=2) and (desaturations["drop"] >= 4.5).all()
    shallow = detect_desaturations(spo2_series(spo2_signal(depth=2.0), CFG), CFG)
    assert shallow.empty  # 2 % is below the 3 % rule
    assert detect_desaturations(spo2_series(spo2_signal(dropout=(0, 600)), CFG), CFG).empty


def test_linking_respects_the_lag_and_window_intervals_merge():
    starts, ends, nadirs = np.array([100.0, 300.0]), np.array([125.0, 340.0]), np.array([145.0, 400.0])
    assert list(confirmed(starts, ends, nadirs, 45)) == [True, False]  # 400 is 60 s after the second event
    assert list(explained(nadirs, starts, ends, 45)) == [True, False]
    s, e = window_intervals(np.arange(0, 300, 30.0), np.arange(30, 330, 30.0), np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1], bool))
    assert list(s) == [30, 150, 240] and list(e) == [90, 180, 300]


def test_scores_reward_events_that_precede_desaturations(tmp_path):
    desaturations = detect_desaturations(spo2_series(spo2_signal(), CFG), CFG)
    starts, ends = np.array([a for a, _ in APNEAS] + [HYPOPNEA[0]]), np.array([b for _, b in APNEAS] + [HYPOPNEA[1]])
    good = desaturation_scores(starts, ends, desaturations, CFG, (0, 600))
    assert good["confirmed"] == 1 and good["explained"] == 1 and good["chance"] < good["confirmed"]
    wrong = desaturation_scores(np.array([10.0, 520.0]), np.array([30.0, 540.0]), desaturations, CFG, (0, 600))
    assert wrong["confirmed"] == 0 and wrong["explained"] == 0
    sleep = desaturation_scores(starts, ends, desaturations, CFG, (0, 600), pd.DataFrame({"end_time": np.arange(30, 630, 30.0), "stage": ["W"] * 20}))
    assert sleep["n_events"] == 0 and sleep["confirmed"] is None  # nothing in sleep


def test_summary_counts_apneas_and_only_confirmed_hypopneas():
    events = pd.DataFrame({"start_time": [100.0, 300.0, 450.0], "end_time": [125.0, 340.0, 470.0], "kind": ["apnea", "hypopnea", "hypopnea"]})
    desaturations = pd.DataFrame({"start_time": [480.0], "nadir_time": [490.0], "end_time": [500.0], "drop": [4.0]})
    summary = oximetry_summary(events, desaturations, 1.0, None, CFG)
    assert summary == {"desaturations": 1, "odi": 1.0, "confirmed_fraction": 1 / 3, "ahi_confirmed": 2.0}
    assert np.isnan(event_desaturation(events, desaturations, 45)[:2]).all() and event_desaturation(events, desaturations, 45)[2] == 4.0


def test_pipeline_writes_desaturations_and_report_scores_detectors(tmp_path, monkeypatch):
    airflow = {"train.edf": signal_with_events().signal, "val.edf": signal_with_events().signal}

    def load(path, channel, start_seconds=0.0, end_seconds=None):
        return spo2_signal() if channel == "SAT" else SignalData(signal=airflow[str(path)], sfreq=SFREQ, start_time=0.0, channel=channel)

    for module in ("psng.pipeline", "psng.examples", "psng.oximetry"):
        monkeypatch.setattr(f"{module}.load_channel", load)
    hypnogram = str(hypnogram_file(tmp_path))
    settings = Settings.model_validate({
        "data": {"channel": "X", "window_seconds": 30, "training": {"file_path": "train.edf", "hypnogram_path": hypnogram},
                 "validation": {"file_path": "val.edf", "hypnogram_path": hypnogram}},
        "features": {"extractor": "respiratory", "reduction": {"type": "pca", "n_components": 3}},
        "models": {"kmeans": {"n_clusters": 3}},
        "run": {"output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs"), "n_jobs": 1, "interactive_plots": False},
    })
    for stage in ("training", "validation"):
        Pipeline(settings, stage).run()
    out = settings.run.output_dir
    assert len(pd.read_csv(out / "training" / "desaturations.csv")) == len(NADIRS)
    assert pd.read_csv(out / "training" / "events.csv")["desaturation"].notna().all()
    summary = json.loads((out / "training" / "events.json").read_text(encoding="utf-8"))
    assert summary["confirmed_fraction"] == 1 and summary["odi"] > 0

    markdown = build_report(out, None, settings)
    scores = pd.read_csv(out / "comparison" / "oximetry.csv")
    assert "SpO2 desaturations" in markdown and {"rules, rolling median", "rules, kalman"} <= set(scores["method"])
    assert (scores.loc[scores["method"] == "rules, rolling median", "confirmed"] == 1).all()
    assert (out / "comparison" / "oximetry.png").exists() and (out / "comparison" / "oximetry_scores.png").exists()
