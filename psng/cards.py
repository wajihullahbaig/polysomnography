"""One card and one examples grid per algorithm, all in the same layout, so methods can be compared at a glance.

Card (comparison/<algorithm>_card.png): the same 10 minutes of night 2 for every algorithm with its decision as a
band, its events next to the rule events, what each of its groups / states looks like over the flow ratio, and an
algorithm-specific matrix. Examples (comparison/<algorithm>_examples.png): signal slices per group for the
clusterers (examples.py), and for the HMM and XGBoost the cases where they agree and disagree with the rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from psng.examples import Recording, _clock, _spread
from psng.events import detect_events, flow_envelope, load_hypnogram, window_stages
from psng.hmm import GaussianHMM, hmm_observation
from psng.oximetry import window_intervals
from psng.plots import EVENT_COLORS, NORMAL_COLOR, STAGE_NAMES, cluster_color, cluster_name, save_card, save_examples_grid
from psng.settings import Settings
from psng.supervised import event_classifier

STATE_COLORS = {"absent flow": EVENT_COLORS["apnea"], "reduced flow": "#e6550d", "normal breathing": NORMAL_COLOR, "large breaths": "#2c7fb8"}
KIND_COLORS = {"normal": NORMAL_COLOR, **EVENT_COLORS}
RESPIRATORY = ("apnea", "hypopnea")


# ------------------------------------------------------------------ helpers
def per_second(time: np.ndarray, starts: np.ndarray, ends: np.ndarray, labels: np.ndarray, default=None) -> np.ndarray:
    """Label of the interval covering each second (intervals sorted and not overlapping), `default` elsewhere."""
    idx = np.searchsorted(np.asarray(starts, float), time, side="right") - 1
    ok = (idx >= 0) & (time < np.asarray(ends, float)[np.clip(idx, 0, None)])
    out = np.full(len(time), default, dtype=object)
    out[ok] = np.asarray(labels, dtype=object)[idx[ok]]
    return out


def density_curves(y: np.ndarray, labels: np.ndarray, colors: dict, names: dict, n_max: int = 4000) -> list[tuple]:
    """Smoothed density of y per label, scaled by the label's share of time (curves add up to the histogram)."""
    grid, out, rng = np.linspace(np.log(0.03), np.log(5), 300), [], np.random.default_rng(0)
    for label in names:
        values = y[labels == label]
        if len(values) < 5 or np.ptp(values) == 0:
            continue
        sample = values if len(values) <= n_max else rng.choice(values, n_max, replace=False)
        share = len(values) / len(y)
        out.append((f"{names[label]}: {share:.0%} of time", colors[label], grid, share * gaussian_kde(sample)(grid)))
    return out


def as_events(starts, ends, kind: str = "event") -> pd.DataFrame:
    return pd.DataFrame({"start_time": np.asarray(starts, float), "end_time": np.asarray(ends, float), "kind": kind})


def in_excerpt(events: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return events[(events["end_time"] > start) & (events["start_time"] < end)]


def runs(time: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-second labels -> (starts, ends, labels) of constant runs."""
    change = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    return time[change], np.r_[time[change[1:]], time[-1] + 1], labels[change]


# ------------------------------------------------------------------ cards
class CardBuilder:
    """Shared night-2 excerpt, observations and rule events; one method per algorithm family."""

    def __init__(self, settings: Settings, recordings: dict[str, Recording], seconds: float = 600.0):
        self.settings, self.recordings, self.out = settings, recordings, settings.run.output_dir
        self.obs = {}
        for stage, rec in recordings.items():
            env = flow_envelope(rec.signal, settings.data.smoothing_seconds)
            _, y = hmm_observation(env["envelope"].to_numpy(float), settings.events, settings.hmm.smoothing_seconds)
            self.obs[stage] = (env["time"].to_numpy(float), y, env)
        self.rules = {s: r.events[r.events["kind"].isin(RESPIRATORY)] for s, r in recordings.items()}
        time, y, _ = self.obs["validation"]
        starts = np.arange(time[0], time[-1] - seconds + 1, 60.0)
        ev = self.rules["validation"]
        # most rule events among stretches whose typical breathing is near normal, so the bands show both states
        typical = np.array([np.exp(np.median(y[(time >= t) & (time < t + seconds)])) for t in starts])
        candidates = starts[(typical >= 0.8) & (typical <= 1.25)] if ((typical >= 0.8) & (typical <= 1.25)).any() else starts
        self.start = max(candidates, key=lambda t: ((ev["start_time"] >= t) & (ev["start_time"] < t + seconds)).sum()) if len(starts) else time[0]
        self.end = self.start + seconds
        self.mask = (time >= self.start) & (time < self.end)

    def excerpt(self, band: tuple, colors: dict, names: dict, events: dict, title: str | None = None) -> dict:
        time, y, _ = self.obs["validation"]
        evs = {"rules": in_excerpt(self.rules["validation"], self.start, self.end), **{k: in_excerpt(v, self.start, self.end) for k, v in events.items()}}
        return {"time": time[self.mask], "ratio": np.exp(y[self.mask]), "band": band, "band_colors": colors, "band_names": names, "events": evs, "band_title": title}

    def save(self, name: str, card: dict) -> Path:
        path = self.out / "comparison" / f"{name}_card.png"
        save_card(card, path)
        return path

    # -------------------------------------------------------------- rules
    def rules_card(self) -> Path:
        time, y, env = self.obs["training"]
        rules = self.recordings["training"].events.sort_values("start_time")
        labels = per_second(time, rules["start_time"], rules["end_time"], rules["kind"], "normal")
        names = {k: k for k in ("normal", "hypopnea", "apnea", "artefact") if (labels == k).any()}
        vtime = self.obs["validation"][0][self.mask]
        vrules = self.recordings["validation"].events.sort_values("start_time")
        band = runs(vtime, per_second(vtime, vrules["start_time"], vrules["end_time"], vrules["kind"], "normal"))
        kalman = detect_events(self.obs["validation"][2], self.settings.events.model_copy(update={"baseline": "kalman"}))
        segment = self.settings.segment("training")
        matrix = self._stage_matrix(rules, load_hypnogram(segment.hypnogram_path) if segment.hypnogram_path else None)
        return self.save("rules", {
            "title": "Threshold rules: an event is breathing below 70 % of its normal level for 10 s or more",
            "subtitle": "The reference every other method is compared with. Below 10 % for 10 s counts as an apnea (a pause), otherwise a hypopnea (shallow breathing).",
            "excerpt": self.excerpt(band, KIND_COLORS, {k: k for k in KIND_COLORS}, {"rules, Kalman level": kalman[kalman["kind"].isin(RESPIRATORY)]}, "rules say"),
            "density": {"y": y, "curves": density_curves(y, labels, KIND_COLORS, names), "title": "What the rule classes look like (night 1)"},
            "matrix": matrix,
        })

    @staticmethod
    def _stage_matrix(events: pd.DataFrame, hypnogram: pd.DataFrame | None) -> dict:
        kinds = [k for k in RESPIRATORY if (events["kind"] == k).any()]
        if hypnogram is None or not kinds:
            return {"values": [[(events["kind"] == k).sum()] for k in kinds] or [[0]], "rows": kinds or ["-"], "cols": ["events"],
                    "title": "Events found (night 1)", "fmt": lambda v: f"{v:.0f}", "vmax": max(len(events), 1)}
        stages = [s for s in ("W", "REM", "N1", "N2", "N3")]
        per_stage = window_stages(events["start_time"].to_numpy(float), events["start_time"].to_numpy(float), hypnogram)
        values = np.array([[np.mean(per_stage[events["kind"].to_numpy() == k] == s) for s in stages] for k in kinds])
        return {"values": values, "rows": kinds, "cols": [STAGE_NAMES[s].split(" (")[0] for s in stages],
                "title": "When do events happen? (share per sleep stage, night 1)", "xlabel": "sleep stage"}

    # -------------------------------------------------------------- clusterers
    def cluster_card(self, model: str) -> Path | None:
        paths = {s: self.out / s / "csvs" / f"{model}.csv" for s in ("training", "validation")}
        if not all(p.exists() for p in paths.values()):
            return None
        train, val = (pd.read_csv(p) for p in paths.values())
        profile = pd.read_csv(self.out / "training" / "csvs" / f"{model}_clusters.csv")
        names = {int(k): str(v) for k, v in zip(profile["cluster_id"], profile["name"])}
        metrics = json.loads((self.out / "training" / "metrics.json").read_text(encoding="utf-8"))
        event_ids = next((m["event_clusters"] for m in metrics if m["algorithm"] == model), [])
        ids = sorted(set(train["cluster_id"]) | set(val["cluster_id"]))
        colors = {k: cluster_color(k, names) for k in ids}
        legend = {k: cluster_name(k, names) for k in ids}

        time, y, _ = self.obs["training"]
        labels = per_second(time, train["window_start_time"], train["window_end_time"], train["cluster_id"])
        win = val[(val["window_end_time"] > self.start) & (val["window_start_time"] < self.end)]
        found = as_events(*window_intervals(val["window_start_time"], val["window_end_time"], val["cluster_id"].isin(event_ids)))
        kinds = ["normal", "hypopnea", "apnea", "artefact"]
        values = np.array([[np.mean(train.loc[train["cluster_id"] == k, "event"] == kind) for kind in kinds] for k in ids])
        event_names = ", ".join(f'"{names[k]}"' for k in event_ids) or "none"
        return self.save(model, {
            "title": f"{model}: groups 30 s windows by how the breathing looks",
            "subtitle": f"Groups are learned on night 1 without labels; the group(s) {event_names} hold mostly events and act as the detector.",
            "excerpt": self.excerpt((win["window_start_time"], win["window_end_time"], win["cluster_id"].to_numpy()), colors, legend,
                                    {f"{model} (event groups)": found}, "group of each window"),
            "density": {"y": y, "curves": density_curves(y, labels, colors, legend), "title": "What each group looks like (night 1)"},
            "matrix": {"values": values, "rows": [cluster_name(k, names) for k in ids], "cols": kinds,
                       "title": "What each group contains (share of its windows, night 1)", "xlabel": "rule label of the window"},
        })

    # -------------------------------------------------------------- HMM
    def hmm_card(self) -> Path | None:
        path = self.out / "training" / "models" / "hmm.pkl"
        if not path.exists():
            return None
        model = GaussianHMM.load(path)
        colors = {k: STATE_COLORS.get(n.split(" #")[0], "#7f7f7f") for k, n in enumerate(model.names_)}
        names = dict(enumerate(model.names_))
        time, y, _ = self.obs["validation"]
        states = model.decode(y)
        band = runs(time[self.mask], states[self.mask])
        hmm = pd.read_csv(self.out / "validation" / "hmm_events.csv")
        weights = np.linalg.matrix_power(model.transitions_.T, 500) @ np.full(model.n_states, 1 / model.n_states)
        grid = np.linspace(np.log(0.03), np.log(5), 300)
        curves = [(f"{n}: typical {model.mean_flow[k]:.2f} x normal, {weights[k]:.0%} of time", colors[k], grid,
                   weights[k] * np.exp(-0.5 * (grid - model.means_[k]) ** 2 / model.vars_[k]) / np.sqrt(2 * np.pi * model.vars_[k]))
                  for k, n in names.items()]
        short = [n.replace(" breathing", "").replace(" breaths", "") for n in model.names_]
        stay = {(k, k): f"\n~{1 / max(1 - p, 1e-9):.0f} s stay" for k, p in enumerate(np.diag(model.transitions_))}
        return self.save("hmm", {
            "title": "Hidden Markov model: learns breathing states and how long each lasts",
            "subtitle": "Learned on night 1 without labels. Each second is judged with its neighbours; orange stretches of 10 s or more become events.",
            "excerpt": self.excerpt(band, colors, {k: f"state: {n}" for k, n in names.items()}, {"HMM": hmm[hmm["kind"].isin(RESPIRATORY)]}, "state of each second"),
            "density": {"y": self.obs["training"][1], "curves": curves, "title": "What each state looks like (night 1)"},
            "matrix": {"values": model.transitions_, "rows": short, "cols": short, "fmt": lambda v: f"{v:.3f}", "extra": stay,
                       "title": "Chance of switching state each second (row to column)", "xlabel": "to", "ylabel": "from"},
        })

    # -------------------------------------------------------------- XGBoost
    def xgboost(self, model_csv: str) -> tuple[pd.DataFrame, np.ndarray] | None:
        paths = [self.out / s / "csvs" / model_csv for s in ("training", "validation")]
        if not all(p.exists() for p in paths):
            return None
        train, val = (pd.read_csv(p) for p in paths)
        return val, event_classifier(train, val)[1]

    def xgboost_card(self, model_csv: str) -> Path | None:
        result = self.xgboost(model_csv)
        if result is None:
            return None
        val, probability = result
        said = np.where(probability >= 0.5, "event", "normal")
        colors, legend = {"event": EVENT_COLORS["hypopnea"], "normal": NORMAL_COLOR}, {"event": "model says event", "normal": "model says normal"}
        time, y, _ = self.obs["validation"]
        labels = per_second(time, val["window_start_time"], val["window_end_time"], said)
        win = (val["window_end_time"] > self.start) & (val["window_start_time"] < self.end)
        actual = np.where(val["event"].isin(RESPIRATORY), "event", "normal")
        values = np.array([[np.mean(said[actual == a] == p) for p in ("normal", "event")] for a in ("normal", "event")])
        counts = {(i, j): f"\n{int(((actual == a) & (said == p)).sum())} windows" for i, a in enumerate(("normal", "event")) for j, p in enumerate(("normal", "event"))}
        return self.save("xgboost", {
            "title": "XGBoost: a supervised model taught by the rules on night 1",
            "subtitle": "It learns from the rule labels of night 1 and is tested on night 2; it shows how much event information the 12 window measurements carry.",
            "excerpt": self.excerpt((val.loc[win, "window_start_time"], val.loc[win, "window_end_time"], said[win.to_numpy()]), colors, legend,
                                    {"XGBoost": as_events(*window_intervals(val["window_start_time"], val["window_end_time"], said == "event"))}, "model decision"),
            "density": {"y": y, "curves": density_curves(y, labels, colors, legend), "title": "What the model's two classes look like (night 2)"},
            "matrix": {"values": values, "rows": ["rules: normal", "rules: event"], "cols": ["model: normal", "model: event"], "extra": counts,
                       "title": "Agreement with the rules on night 2 (share of each row)"},
        })


# ------------------------------------------------------------------ agreement examples
def agreement_examples(recordings: dict[str, Recording], found: dict[str, pd.DataFrame], algorithm: str, n: int = 3, context: float = 15.0) -> dict:
    """Rows: both find an event / only the algorithm / only the rules / neither. Grey = the algorithm's event, colour = rule events."""
    rows = {f"both {algorithm} and rules": {}, f"only {algorithm}": {}, "only the rules": {}, "neither (normal)": {}}
    for stage, events in found.items():
        rec = recordings[stage]
        rules = rec.events[rec.events["kind"].isin(RESPIRATORY)]

        def overlaps(s, e, other):
            return ((other["start_time"] < e) & (other["end_time"] > s)).any()

        hit = np.array([overlaps(r.start_time, r.end_time, rules) for r in events.itertuples()], bool)
        caught = np.array([overlaps(r.start_time, r.end_time, events) for r in rules.itertuples()], bool)
        for key, part in ((f"both {algorithm} and rules", events[hit]), (f"only {algorithm}", events[~hit])):
            rows[key][stage] = [rec.slice(r.start_time - context, r.end_time + context, f"{_clock(r.start_time)}  {r.end_time - r.start_time:.0f}s",
                                          window=(r.start_time, r.end_time)) for r in part.iloc[_spread(len(part), n)].itertuples()]
        missed = rules[~caught]
        rows["only the rules"][stage] = [rec.slice(r.start_time - context, r.end_time + context, f"{_clock(r.start_time)}  {r.duration:.0f}s  {r.kind}")
                                         for r in missed.iloc[_spread(len(missed), n)].itertuples()]
        starts = np.arange(rec.signal.start_time, rec.signal.start_time + rec.signal.duration - 60, 60.0)
        busy = np.zeros(len(starts), bool)
        for r in pd.concat([rec.events, events]).itertuples():  # artefacts (sensor off) are not normal breathing either
            busy |= (starts < r.end_time) & (starts + 60 > r.start_time)
        free = starts[~busy]
        rows["neither (normal)"][stage] = [rec.slice(t, t + 60, f"{_clock(t)}  no event") for t in free[_spread(len(free), n)]]
    return rows


def build_cards(settings: Settings, recordings: dict[str, Recording], models: list[str]) -> list[Path]:
    """Every card and the HMM / XGBoost examples; needs both stages."""
    if not {"training", "validation"} <= set(recordings):
        return []
    builder = CardBuilder(settings, recordings)
    written = [builder.rules_card(), builder.hmm_card(), *(builder.cluster_card(m) for m in models)]
    if models:
        written.append(builder.xgboost_card(f"{models[0]}.csv"))
    comparison = settings.run.output_dir / "comparison"
    hmm = {s: pd.read_csv(settings.run.output_dir / s / "hmm_events.csv") for s in recordings if (settings.run.output_dir / s / "hmm_events.csv").exists()}
    if hmm:
        save_examples_grid(agreement_examples(recordings, {s: e[e["kind"].isin(RESPIRATORY)] for s, e in hmm.items()}, "HMM"),
                           "HMM: where it agrees and disagrees with the rules (grey = HMM event, orange / red = rule event)", comparison / "hmm_examples.png")
        written.append(comparison / "hmm_examples.png")
    result = builder.xgboost(f"{models[0]}.csv") if models else None
    if result is not None:
        val, probability = result
        found = as_events(*window_intervals(val["window_start_time"], val["window_end_time"], probability >= 0.5))
        save_examples_grid(agreement_examples(recordings, {"validation": found}, "XGBoost"),
                           "XGBoost on night 2: where it agrees and disagrees with the rules (grey = model event, orange / red = rule event)",
                           comparison / "xgboost_examples.png")
        written.append(comparison / "xgboost_examples.png")
    return [p for p in written if p is not None]
