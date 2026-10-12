"""SpO2 desaturations: an airflow-independent reference for breathing events.

Every other reference in the pipeline comes from the airflow itself, so methods can only agree or disagree with
the rule detector. Oxygen saturation is a second, independent channel: an obstructed breath lowers SpO2 about
15-30 s after the event ends (circulation delay). Here

* a **desaturation** is SpO2 at least `drop_percent` below its maximum of the preceding `lookback_seconds`
  for at least `min_seconds` (oxygen desaturation index, ODI, counts them per hour of sleep);
* an airflow event is **confirmed** when a desaturation nadir falls between its start and `lag_seconds` after
  its end (AASM 1A hypopnea: >= 30 % flow drop with >= 3 % desaturation; arousals are not checked);
* any detector (rules, clusters, a classifier) is scored by the share of its events that are confirmed, the
  share of sleep desaturations it explains, and the confirmed share expected by chance (desaturations shifted
  circularly in time), so a detector that flags everything cannot look good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psng.data import SLEEP_STAGES, SignalData
from psng.edf import load_channel
from psng.events import _runs, window_stages
from psng.settings import OximetrySettings, Settings

DESATURATION_COLUMNS = ["start_time", "nadir_time", "end_time", "drop"]
SLEEP = [s for s in SLEEP_STAGES if s != "W"]


def load_spo2(settings: Settings, stage: str) -> pd.DataFrame | None:
    """1 Hz SpO2 of a stage's segment, or None when the oximetry channel is disabled or missing."""
    cfg, segment = settings.oximetry, settings.segment(stage)
    if cfg.channel is None:
        return None
    try:
        return spo2_series(load_channel(segment.file_path, cfg.channel, segment.start_seconds, segment.end_seconds), cfg)
    except (ValueError, OSError):  # channel or file missing
        return None


def spo2_series(signal: SignalData, cfg: OximetrySettings) -> pd.DataFrame:
    """1 Hz SpO2 (median per second): values outside `valid_range` (probe off) are NaN, gaps up to `max_gap_seconds` interpolated."""
    sf = int(round(signal.sfreq))
    seconds = len(signal.signal) // sf
    spo2 = np.median(np.asarray(signal.signal[: seconds * sf], float).reshape(seconds, sf), axis=1)
    low, high = cfg.valid_range
    spo2[(spo2 < low) | (spo2 > high)] = np.nan
    spo2 = pd.Series(spo2).interpolate(limit=int(cfg.max_gap_seconds), limit_area="inside").to_numpy()
    return pd.DataFrame({"time": signal.start_time + np.arange(seconds), "spo2": spo2})


def detect_desaturations(spo2: pd.DataFrame, cfg: OximetrySettings) -> pd.DataFrame:
    """Desaturations as rows (start_time, nadir_time, end_time, drop); `drop` is the deepest fall below the preceding maximum, in %."""
    level = spo2["spo2"].rolling(3, center=True, min_periods=1).median()  # 1 % quantised sensor: drop single-sample flicker
    drop = (level.rolling(int(cfg.lookback_seconds), min_periods=1).max() - level).to_numpy()
    time, level = spo2["time"].to_numpy(float), level.to_numpy()
    rows = []
    for start, stop in _runs(np.nan_to_num(drop) >= cfg.drop_percent):
        if stop - start >= cfg.min_seconds:
            nadir = start + int(np.nanargmin(level[start:stop]))
            rows.append((time[start], time[nadir], time[stop - 1] + 1, float(drop[start:stop].max())))
    return pd.DataFrame(rows, columns=DESATURATION_COLUMNS)


# ------------------------------------------------------------------ linking events and desaturations
def confirmed(starts: np.ndarray, ends: np.ndarray, nadirs: np.ndarray, lag_seconds: float) -> np.ndarray:
    """Per event: is there a desaturation nadir in [start, end + lag]?"""
    nadirs = np.sort(np.asarray(nadirs, float))
    first = np.searchsorted(nadirs, np.asarray(starts, float), side="left")
    last = np.searchsorted(nadirs, np.asarray(ends, float) + lag_seconds, side="right")
    return last > first


def explained(nadirs: np.ndarray, starts: np.ndarray, ends: np.ndarray, lag_seconds: float) -> np.ndarray:
    """Per desaturation: does its nadir fall in [start, end + lag] of any event?"""
    nadirs = np.asarray(nadirs, float)
    hit = np.zeros(len(nadirs), bool)
    for s, e in zip(np.asarray(starts, float), np.asarray(ends, float)):
        hit |= (nadirs >= s) & (nadirs <= e + lag_seconds)
    return hit


def event_desaturation(events: pd.DataFrame, desaturations: pd.DataFrame, lag_seconds: float) -> np.ndarray:
    """Deepest desaturation drop (%) linked to each event, NaN when unconfirmed (the `desaturation` column of events.csv)."""
    out = np.full(len(events), np.nan)
    nadirs, drops = desaturations["nadir_time"].to_numpy(float), desaturations["drop"].to_numpy(float)
    for i, (s, e) in enumerate(zip(events["start_time"].to_numpy(float), events["end_time"].to_numpy(float))):
        m = (nadirs >= s) & (nadirs <= e + lag_seconds)
        if m.any():
            out[i] = drops[m].max()
    return out


def window_intervals(starts: np.ndarray, ends: np.ndarray, flagged: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive flagged windows merged into (start, end) event intervals, so window detectors are scored like event detectors."""
    starts, ends, flagged = np.asarray(starts, float), np.asarray(ends, float), np.asarray(flagged, bool)
    pairs = _runs(flagged)
    return np.array([starts[a] for a, _ in pairs]), np.array([ends[b - 1] for _, b in pairs])


# ------------------------------------------------------------------ scores
def desaturation_scores(starts: np.ndarray, ends: np.ndarray, desaturations: pd.DataFrame, cfg: OximetrySettings,
                        span: tuple[float, float], hypnogram: pd.DataFrame | None = None, n_shifts: int = 200, seed: int = 0) -> dict:
    """Score detected events against desaturations, in sleep only when a hypnogram is given.

    confirmed  share of events followed by a desaturation (precision against SpO2)
    chance     confirmed share when the desaturations are shifted circularly by random offsets (>= 10 min)
    lift       confirmed / chance
    explained  share of desaturations that follow an event (recall against SpO2)
    f1         harmonic mean of confirmed and explained
    """
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    nadirs = desaturations["nadir_time"].to_numpy(float)
    if hypnogram is not None and len(hypnogram):
        starts, ends = (v[np.isin(window_stages(starts, starts, hypnogram), SLEEP)] for v in (starts, ends))
        nadirs = nadirs[np.isin(window_stages(nadirs, nadirs, hypnogram), SLEEP)]
    if len(starts) == 0 or len(nadirs) == 0:
        return {"n_events": len(starts), "confirmed": None, "chance": None, "lift": None, "explained": None, "f1": None}
    precision = float(confirmed(starts, ends, nadirs, cfg.lag_seconds).mean())
    recall = float(explained(nadirs, starts, ends, cfg.lag_seconds).mean())
    t0, length = span[0], span[1] - span[0]
    margin = min(600.0, length / 4)  # shifts of at least 10 min (short recordings: a quarter of their length)
    offsets = np.random.default_rng(seed).uniform(margin, length - margin, n_shifts)
    chance = float(np.mean([confirmed(starts, ends, t0 + (nadirs - t0 + o) % length, cfg.lag_seconds).mean() for o in offsets]))
    return {"n_events": len(starts), "confirmed": precision, "chance": chance, "lift": precision / chance if chance > 0 else None,
            "explained": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0}


def oximetry_summary(events: pd.DataFrame, desaturations: pd.DataFrame, sleep_hours: float | None, hypnogram: pd.DataFrame | None,
                     cfg: OximetrySettings) -> dict:
    """Night-level numbers: ODI (desaturations per hour of sleep), share of airflow events confirmed, AHI with confirmed hypopneas."""
    respiratory = events[events["kind"].isin(("apnea", "hypopnea"))]
    nadirs = desaturations["nadir_time"].to_numpy(float)
    keep = np.ones(len(respiratory), bool)
    if hypnogram is not None and len(hypnogram):
        nadirs = nadirs[np.isin(window_stages(nadirs, nadirs, hypnogram), SLEEP)]
        keep = np.isin(window_stages(respiratory["start_time"].to_numpy(float), respiratory["start_time"].to_numpy(float), hypnogram), SLEEP)
    linked = confirmed(respiratory["start_time"], respiratory["end_time"], nadirs, cfg.lag_seconds)
    counted = keep & ((respiratory["kind"] == "apnea").to_numpy() | linked)  # apneas count without a desaturation (AASM)
    per_hour = (lambda n: n / sleep_hours) if sleep_hours else (lambda n: None)
    return {"desaturations": int(len(desaturations)), "odi": per_hour(len(nadirs)),
            "confirmed_fraction": float(linked[keep].mean()) if keep.any() else None, "ahi_confirmed": per_hour(int(counted.sum()))}
