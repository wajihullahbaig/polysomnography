"""Example time slices per event kind and per cluster, training next to validation.

The signal is band-passed like the windows and expressed in baseline units (1.0 = typical breathing
amplitude of that recording), so a flat apnea and a normal breath look the same in both recordings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from psng.data import SignalData
from psng.edf import load_channel
from psng.events import EVENT_COLUMNS
from psng.plots import save_examples_grid
from psng.settings import Settings
from psng.windows import WindowPreprocessor

STAGES = ("training", "validation")
EVENT_ROWS = ("apnea", "hypopnea", "artefact", "normal")


class Recording:
    """One stage's signal plus its events and baseline, ready to cut slices from."""

    def __init__(self, settings: Settings, stage: str):
        segment = settings.segment(stage)
        self.stage = stage
        self.signal: SignalData = load_channel(segment.file_path, settings.data.channel, segment.start_seconds, segment.end_seconds)
        stage_dir = settings.run.output_dir / stage
        self.events = pd.read_csv(stage_dir / "events.csv") if (stage_dir / "events.csv").exists() else pd.DataFrame(columns=EVENT_COLUMNS)
        summary = json.loads((stage_dir / "events.json").read_text(encoding="utf-8")) if (stage_dir / "events.json").exists() else {}
        self.baseline = float(summary.get("baseline_amplitude") or 1.0)
        self.filter = WindowPreprocessor(self.signal.sfreq, settings.data.smoothing_seconds).filter

    def slice(self, start: float, end: float, label: str, window: tuple[float, float] | None = None) -> dict:
        s = self.signal
        start, end = max(start, s.start_time), min(end, s.start_time + s.duration)
        a, b = int((start - s.start_time) * s.sfreq), int((end - s.start_time) * s.sfreq)
        y = self.filter(s.signal[a:b]) / max(self.baseline, 1e-12)
        t = start + np.arange(len(y)) / s.sfreq
        spans = [(r.start_time, r.end_time, r.kind) for r in self.events.itertuples() if r.end_time > start and r.start_time < end]
        if window is not None:
            spans.insert(0, (window[0], window[1], "window"))
        return {"t": t, "y": y, "spans": spans, "label": label}


def _spread(n_available: int, n: int) -> np.ndarray:
    """Indices spread evenly over the available items (deterministic, covers the whole night)."""
    if n_available <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, n_available - 1, min(n, n_available)).round().astype(int))


def event_examples(recordings: dict[str, Recording], n: int = 3, context_seconds: float = 15.0, max_seconds: float = 150.0) -> dict:
    """Per event kind: `n` events per stage with context on both sides; `normal` rows show windows without events."""
    out: dict[str, dict[str, list[dict]]] = {kind: {} for kind in EVENT_ROWS}
    for stage, rec in recordings.items():
        for kind in EVENT_ROWS:
            if kind == "normal":
                out[kind][stage] = _normal_examples(rec, n)
                continue
            rows = rec.events[rec.events["kind"] == kind].sort_values("start_time").reset_index(drop=True)
            out[kind][stage] = [
                rec.slice(r.start_time - context_seconds, min(r.end_time, r.start_time + max_seconds) + context_seconds,
                          f"{_clock(r.start_time)}  {r.duration:.0f}s  flow {r.min_flow:.0%}")
                for r in rows.iloc[_spread(len(rows), n)].itertuples()
            ]
    return out


def _normal_examples(rec: Recording, n: int, seconds: float = 60.0) -> list[dict]:
    s = rec.signal
    starts = np.arange(s.start_time, s.start_time + s.duration - seconds, seconds)
    busy = np.zeros(len(starts), dtype=bool)
    for r in rec.events.itertuples():
        busy |= (starts < r.end_time) & (starts + seconds > r.start_time)
    free = starts[~busy]
    return [rec.slice(t, t + seconds, f"{_clock(t)}  no event") for t in free[_spread(len(free), n)]]


def cluster_examples(recordings: dict[str, Recording], frames: dict[str, pd.DataFrame], names: dict[int, str], n: int = 3,
                     context_seconds: float = 15.0) -> dict:
    """Per cluster: `n` windows per stage spread over the night, window shaded grey, events shaded by kind."""
    ids = sorted({int(k) for f in frames.values() for k in f["cluster_id"].unique()})
    out: dict[str, dict[str, list[dict]]] = {}
    for k in ids:
        row = names.get(k, "noise" if k < 0 else f"cluster {k}")
        row = row if k < 0 else f"{k}: {row}"
        out[row] = {}
        for stage, frame in frames.items():
            rec = recordings[stage]
            windows = frame[frame["cluster_id"] == k].reset_index(drop=True)
            out[row][stage] = [
                rec.slice(w.window_start_time - context_seconds, w.window_end_time + context_seconds,
                          f"{_clock(w.window_start_time)}  {w.event}" + (f"  {w.stage}" if "stage" in frame.columns else ""),
                          window=(w.window_start_time, w.window_end_time))
                for w in windows.iloc[_spread(len(windows), n)].itertuples()
            ]
    return out


def load_recordings(settings: Settings) -> dict[str, Recording]:
    output_dir = settings.run.output_dir
    return {stage: Recording(settings, stage) for stage in STAGES if (output_dir / stage / "events.csv").exists()}


def example_figures(settings: Settings, models: list[str], n: int = 3, recordings: dict[str, Recording] | None = None) -> list[Path]:
    """Write comparison/event_examples.png and comparison/<model>_examples.png; returns the written paths."""
    output_dir = settings.run.output_dir
    recordings = load_recordings(settings) if recordings is None else recordings
    if not recordings:
        return []
    comparison = output_dir / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    written = []
    path = comparison / "event_examples.png"
    save_examples_grid(event_examples(recordings, n), f"Example airflow slices per detected event kind ({settings.data.channel}, baseline units)", path)
    written.append(path)
    for model in models:
        frames = {stage: pd.read_csv(output_dir / stage / "csvs" / f"{model}.csv") for stage in recordings
                  if (output_dir / stage / "csvs" / f"{model}.csv").exists()}
        if not frames:
            continue
        names = {}
        profile = output_dir / "training" / "csvs" / f"{model}_clusters.csv"
        if profile.exists():
            p = pd.read_csv(profile)
            names = dict(zip(p["cluster_id"].astype(int), p["name"].astype(str)))
        path = comparison / f"{model}_examples.png"
        save_examples_grid(cluster_examples(recordings, frames, names, n), f"{model}: example windows per cluster (window shaded grey, events by kind)", path)
        written.append(path)
    return written


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
