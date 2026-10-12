"""Rule-based airflow event detection, hypnogram loading and label-based cluster scores.

Events are found on a 1 Hz RMS envelope of the band-passed airflow, expressed relative to a rolling
baseline (the local typical breathing amplitude):

* apnea     - flow below `apnea_ratio` of baseline for at least `min_seconds` (AASM: >= 90 % drop, >= 10 s)
* hypopnea  - flow below `hypopnea_ratio` of baseline for at least `min_seconds` (AASM: >= 30 % drop, >= 10 s)
* artefact  - a low-flow stretch longer than `max_seconds` (sensor off, disconnection)

Flow-only rules cannot check the desaturation / arousal criteria of a clinical scoring, so the
events are a reference for the clusters rather than a diagnosis. The hypnogram is used for the
Apnea-Hypopnea Index (events per hour of sleep) and as a second external reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import normalized_mutual_info_score

from psng.data import EVENT_KINDS, SLEEP_STAGES, EventSummary, SignalData
from psng.kalman import kalman_smooth
from psng.settings import EventSettings
from psng.windows import RESPIRATORY_BAND_HZ, interpolate_nans

EVENT_COLUMNS = ["start_time", "end_time", "duration", "kind", "min_flow"]
_PRIORITY = ("apnea", "hypopnea", "artefact")  # when a window overlaps several kinds


# ------------------------------------------------------------------ envelope
def flow_envelope(signal: SignalData, smoothing_seconds: float = 0.5, band_hz: tuple[float, float] = RESPIRATORY_BAND_HZ) -> pd.DataFrame:
    """1 Hz RMS envelope of the band-passed signal: columns `time` (second start) and `envelope`."""
    sf = int(round(signal.sfreq))
    x = interpolate_nans(np.asarray(signal.signal, dtype=float))
    n = int(smoothing_seconds * signal.sfreq) | 1
    if smoothing_seconds > 0:
        x = uniform_filter1d(x, n, mode="nearest")
    x = x - x.mean()
    x = sps.sosfiltfilt(sps.butter(3, band_hz, btype="band", fs=signal.sfreq, output="sos"), x)
    seconds = len(x) // sf
    rms = np.sqrt(np.mean(x[: seconds * sf].reshape(seconds, sf) ** 2, axis=1))
    return pd.DataFrame({"time": signal.start_time + np.arange(seconds), "envelope": rms})


def recording_baseline(envelope: pd.DataFrame) -> float:
    """Typical breathing amplitude of the recording (median of the envelope)."""
    return float(np.median(envelope["envelope"])) if len(envelope) else 0.0


def rolling_baseline(envelope: np.ndarray, seconds: float, floor: float) -> np.ndarray:
    """Centred rolling median of the envelope, never below `floor` so long flat stretches stay detectable."""
    window = max(3, int(seconds)) | 1
    base = pd.Series(envelope).rolling(window, center=True, min_periods=1).median().to_numpy()
    return np.maximum(base, floor)


def baseline_variants(envelope: np.ndarray, cfg: EventSettings) -> dict[str, np.ndarray]:
    """Both trackers (rolling median and Kalman) with the configured parameters, for plots and comparisons."""
    return {name: local_baseline(envelope, cfg.model_copy(update={"baseline": name})) for name in ("rolling_median", "kalman")}


def local_baseline(envelope: np.ndarray, cfg: EventSettings) -> np.ndarray:
    """Local breathing amplitude: rolling median or Kalman-smoothed level, floored at half the night median."""
    floor = 0.5 * float(np.median(envelope)) if len(envelope) else 0.0
    if cfg.baseline == "kalman":
        return np.maximum(kalman_smooth(envelope, cfg.kalman_process_ratio), floor)
    return rolling_baseline(envelope, cfg.baseline_seconds, floor)


# ------------------------------------------------------------------ detection
def detect_events(envelope: pd.DataFrame, cfg: EventSettings) -> pd.DataFrame:
    """Events as rows (start_time, end_time, duration, kind, min_flow); `min_flow` is the lowest flow / baseline ratio."""
    env = envelope["envelope"].to_numpy(float)
    time = envelope["time"].to_numpy(float)
    if len(env) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    baseline = local_baseline(env, cfg)
    ratio = env / np.maximum(baseline, 1e-12)
    rows = []
    for start, stop in _runs(ratio < cfg.hypopnea_ratio):
        duration = float(stop - start)  # 1 Hz samples
        if duration < cfg.min_seconds:
            continue
        if duration > cfg.max_seconds:
            kind = "artefact"
        else:
            deepest = max((b - a for a, b in _runs(ratio[start:stop] < cfg.apnea_ratio)), default=0)
            kind = "apnea" if deepest >= cfg.min_seconds else "hypopnea"
        rows.append((time[start], time[start] + duration, duration, kind, float(ratio[start:stop].min())))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of consecutive True values."""
    padded = np.r_[False, mask.astype(bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


# ------------------------------------------------------------------ window labels
def window_event_labels(starts: np.ndarray, ends: np.ndarray, events: pd.DataFrame, min_overlap_seconds: float) -> np.ndarray:
    """Event kind per window: the highest-priority kind it overlaps for at least `min_overlap_seconds`, else `normal`."""
    labels = np.full(len(starts), "normal", dtype=object)
    if len(events) == 0 or len(starts) == 0:
        return labels
    for kind in reversed(_PRIORITY):  # lowest priority first, higher ones overwrite
        rows = events[events["kind"] == kind]
        if rows.empty:
            continue
        overlap = np.zeros(len(starts))
        for s, e in zip(rows["start_time"].to_numpy(float), rows["end_time"].to_numpy(float)):
            overlap += np.clip(np.minimum(ends, e) - np.maximum(starts, s), 0, None)
        labels[overlap >= max(min_overlap_seconds, 1e-9)] = kind
    return labels


def load_hypnogram(path: str | Path) -> pd.DataFrame:
    """Tab-separated `<epoch end seconds> <clock> <stage> <code>` rows (one per 30 s epoch) -> columns end_time, stage."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 2], names=["end_time", "stage"], engine="python", comment="#")
    frame = frame[pd.to_numeric(frame["end_time"], errors="coerce").notna()]
    frame["end_time"] = frame["end_time"].astype(float)
    frame["stage"] = frame["stage"].astype(str).str.strip().str.upper().replace({"R": "REM", "S1": "N1", "S2": "N2", "S3": "N3", "S4": "N3"})
    return frame.sort_values("end_time").reset_index(drop=True)


def window_stages(starts: np.ndarray, ends: np.ndarray, hypnogram: pd.DataFrame) -> np.ndarray:
    """Sleep stage of the epoch containing each window midpoint (`?` when the hypnogram does not cover it)."""
    mid = (np.asarray(starts, float) + np.asarray(ends, float)) / 2
    ends_h = hypnogram["end_time"].to_numpy(float)
    idx = np.searchsorted(ends_h, mid, side="right")
    stages = np.full(len(mid), "?", dtype=object)
    ok = idx < len(ends_h)
    stages[ok] = hypnogram["stage"].to_numpy(object)[idx[ok]]
    return stages


# ------------------------------------------------------------------ summaries and scores
def event_summary(stage: str, events: pd.DataFrame, event_labels: np.ndarray, stages: np.ndarray | None,
                  duration_seconds: float, baseline_amplitude: float, hypnogram: pd.DataFrame | None = None) -> EventSummary:
    counts = events["kind"].value_counts() if len(events) else pd.Series(dtype=int)
    n_events = int(counts.get("apnea", 0) + counts.get("hypopnea", 0))
    hours = duration_seconds / 3600
    sleep_hours = ahi = None
    if hypnogram is not None and len(hypnogram):
        epoch = float(np.median(np.diff(hypnogram["end_time"]))) if len(hypnogram) > 1 else 30.0
        asleep = hypnogram[(hypnogram["stage"] != "W") & (hypnogram["end_time"] <= duration_seconds + epoch)]
        sleep_hours = len(asleep) * epoch / 3600
        if len(events) and sleep_hours > 0:
            respiratory = events[events["kind"].isin(("apnea", "hypopnea"))]
            in_sleep = window_stages(respiratory["start_time"].to_numpy(float), respiratory["start_time"].to_numpy(float), hypnogram)
            ahi = float(np.isin(in_sleep, [s for s in SLEEP_STAGES if s != "W"]).sum() / sleep_hours)
    return EventSummary(
        stage=stage, duration_hours=hours, sleep_hours=sleep_hours, n_windows=len(event_labels),
        apnea=int(counts.get("apnea", 0)), hypopnea=int(counts.get("hypopnea", 0)), artefact=int(counts.get("artefact", 0)),
        events_per_hour=n_events / hours if hours > 0 else 0.0, ahi=ahi,
        event_window_fraction=float(np.isin(event_labels, ["apnea", "hypopnea"]).mean()) if len(event_labels) else 0.0,
        baseline_amplitude=baseline_amplitude,
    )


def reference_scores(labels: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    """Normalised mutual information and purity of the clusters against reference labels (noise excluded)."""
    labels, reference = np.asarray(labels), np.asarray(reference)
    mask = labels >= 0
    if mask.sum() < 2:
        return {"nmi": None, "purity": None}
    table = pd.crosstab(labels[mask], reference[mask])
    return {"nmi": float(normalized_mutual_info_score(reference[mask], labels[mask])),
            "purity": float(table.max(axis=1).sum() / table.to_numpy().sum())}


def event_detection_scores(labels: np.ndarray, event_labels: np.ndarray) -> dict:
    """Treat every cluster whose majority is apnea / hypopnea as an event detector; precision, recall and F1 on windows."""
    labels, is_event = np.asarray(labels), np.isin(event_labels, ["apnea", "hypopnea"])
    ids = np.unique(labels[labels >= 0])
    event_clusters = [int(k) for k in ids if is_event[labels == k].mean() > 0.5]
    predicted = np.isin(labels, event_clusters)
    tp = int((predicted & is_event).sum())
    precision = tp / predicted.sum() if predicted.any() else 0.0
    recall = tp / is_event.sum() if is_event.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"event_precision": precision, "event_recall": recall, "event_f1": f1, "event_clusters": event_clusters}


def cluster_profile(labels: np.ndarray, event_labels: np.ndarray, stages: np.ndarray | None, features: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster: size, event composition, stage composition and mean raw features."""
    frame = pd.DataFrame({"cluster_id": np.asarray(labels)})
    out = frame.groupby("cluster_id").size().rename("n_windows").to_frame()
    out["share"] = out["n_windows"] / len(frame)
    kinds = pd.crosstab(frame["cluster_id"], pd.Categorical(event_labels, categories=EVENT_KINDS), dropna=False)
    out["event_rate"] = (kinds["apnea"] + kinds["hypopnea"]) / out["n_windows"]
    out = out.join(kinds.add_prefix("n_"))
    if stages is not None:
        out = out.join(pd.crosstab(frame["cluster_id"], pd.Categorical(stages, categories=SLEEP_STAGES), dropna=False).add_prefix("n_"))
    means = features.groupby(frame["cluster_id"].to_numpy()).mean().add_prefix("mean_")
    return out.join(means).reset_index()


# ------------------------------------------------------------------ cluster names
def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Human-readable name per cluster from its event composition and mean features (needs `cluster_profile` output).

    Event-dominated clusters are named after the dominant event; the rest by airflow amplitude, breathing
    rate and regularity (available with the `respiratory` extractor), with the dominant sleep stage appended
    when a hypnogram covers more than 60 % of the cluster. Duplicate names get a numeric suffix.
    """
    names: dict[int, str] = {}
    for row in profile.itertuples(index=False):
        k = int(row.cluster_id)
        if k < 0:
            names[k] = "noise"
            continue
        n = max(int(row.n_windows), 1)
        shares = {kind: getattr(row, f"n_{kind}", 0) / n for kind in ("apnea", "hypopnea", "artefact")}
        if shares["apnea"] + shares["hypopnea"] > 0.5:
            name = "apnea" if shares["apnea"] >= shares["hypopnea"] else "hypopnea"
        elif shares["artefact"] > 0.5:
            name = "artefact / sensor off"
        else:
            name = _breathing_name(row)
            if shares["apnea"] + shares["hypopnea"] > 0.25:
                name += " with events"
        stage_counts = {s: getattr(row, f"n_{s}", 0) for s in SLEEP_STAGES if hasattr(row, f"n_{s}")}
        if stage_counts and sum(stage_counts.values()) > 0:
            stage, count = max(stage_counts.items(), key=lambda kv: kv[1])
            if count / n > 0.6:
                name += f", mostly {stage}"
        names[k] = name
    seen: dict[str, int] = {}
    for k in sorted(names):
        seen[names[k]] = seen.get(names[k], 0) + 1
        if seen[names[k]] > 1:
            names[k] = f"{names[k]} #{seen[names[k]]}"
    return names


def _breathing_name(row) -> str:
    amplitude = getattr(row, "mean_amplitude", None)
    rate = getattr(row, "mean_breathing_rate", None)
    regularity = getattr(row, "mean_breathing_regularity", None)
    parts = []
    if amplitude is not None and np.isfinite(amplitude):
        if amplitude < 0.5:
            parts.append("reduced airflow")
        elif amplitude > 1.5:
            parts.append("large breaths")
    if rate is not None and np.isfinite(rate):
        if rate < 10:
            parts.append("slow breathing")
        elif rate > 22:
            parts.append("fast breathing")
    if regularity is not None and np.isfinite(regularity) and regularity > 0.4:
        parts.append("irregular rhythm")
    return ", ".join(parts) if parts else "regular breathing"
