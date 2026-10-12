"""Training / validation orchestration with resumable checkpoints.

Training:   EDF -> windows -> features (cached per chunk) -> fit transformer -> fit models -> outputs
Validation: EDF -> windows -> features (cached per chunk) -> load transformer -> load models -> outputs

Both stages also run the rule-based airflow event detector on the recording and, when a hypnogram
is configured, read the sleep stages; these label every window and score the clusters. When the
recording has an SpO2 channel, desaturations are detected and linked to the events (oximetry.py).
A hidden Markov model on the flow ratio (hmm.py) is fitted on training and decodes both stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from psng.clustering import Clusterer, make_clusterer
from psng.data import META_COLUMNS, ClusterMetrics, SignalData, feature_matrix
from psng.edf import load_channel
from psng.events import (baseline_variants, cluster_profile, detect_events, event_summary, flow_envelope, load_hypnogram, name_clusters,
                         recording_baseline, window_event_labels, window_stages)
from psng.features import FeatureExtractor, make_extractor
from psng.hmm import GaussianHMM, fit_hmm, hmm_events
from psng.log import get_logger
from psng.metrics import evaluate, format_table, write_metrics
from psng.oximetry import DESATURATION_COLUMNS, detect_desaturations, event_desaturation, oximetry_summary, spo2_series
from psng.plots import save_events_overview, save_interactive, save_scatter, save_timeline
from psng.settings import Settings, Stage
from psng.transform import FeatureTransformer
from psng.windows import WindowPreprocessor, window_starts


def window_features(window: np.ndarray, preprocess: WindowPreprocessor, extractor: FeatureExtractor) -> dict[str, float]:
    return extractor.extract(preprocess(window))


class Paths:
    def __init__(self, output_dir: Path, stage: Stage):
        self.models = output_dir / "training" / "models"  # models are always read from training
        self.cache = output_dir / "cache"
        self.stage_dir = output_dir / stage
        self.csvs = self.stage_dir / "csvs"
        self.plots = self.stage_dir / "plots"
        self.metrics = self.stage_dir / "metrics.json"
        self.events = self.stage_dir / "events.csv"
        self.event_summary = self.stage_dir / "events.json"
        self.desaturations = self.stage_dir / "desaturations.csv"
        self.hmm_events = self.stage_dir / "hmm_events.csv"
        self.hmm_summary = self.stage_dir / "hmm_events.json"
        for folder in (self.models, self.cache, self.csvs, self.plots):
            folder.mkdir(parents=True, exist_ok=True)


@dataclass
class Prepared:
    """Everything a stage computes before any clusterer runs."""

    signal: SignalData
    envelope: pd.DataFrame
    baseline: float
    events: pd.DataFrame
    hypnogram: pd.DataFrame | None
    frame: pd.DataFrame  # window times, raw features, `event` and (optionally) `stage` labels
    X: pd.DataFrame  # transformed features the clusterers see

    @property
    def event_labels(self) -> np.ndarray:
        return self.frame["event"].to_numpy(object)

    @property
    def stages(self) -> np.ndarray | None:
        return self.frame["stage"].to_numpy(object) if "stage" in self.frame.columns else None


class Pipeline:
    def __init__(self, settings: Settings, stage: Stage):
        self.settings = settings
        self.stage = stage
        self.resume = settings.run.resume
        self.paths = Paths(settings.run.output_dir, stage)
        self.log = get_logger(f"psng.{stage}", settings.run.log_dir / f"{stage}.log", settings.run.log_level)

    def run(self) -> list[ClusterMetrics]:
        prepared = self.prepare()
        metrics = [self.cluster(name, params, prepared) for name, params in self.settings.models.items()]
        write_metrics(metrics, self.paths.metrics)
        self.log.info("metrics written to %s\n%s", self.paths.metrics, format_table(metrics))
        return metrics

    def prepare(self) -> Prepared:
        segment = self.settings.segment(self.stage)
        cfg = self.settings.data
        self.log.info("stage=%s file=%s channel=%s segment=%s-%ss resume=%s",
                      self.stage, segment.file_path, cfg.channel, segment.start_seconds, segment.end_seconds, self.resume)
        signal = load_channel(segment.file_path, cfg.channel, segment.start_seconds, segment.end_seconds)
        self.log.info("loaded %.1f s at %g Hz", signal.duration, signal.sfreq)

        envelope = flow_envelope(signal, cfg.smoothing_seconds)
        baseline = recording_baseline(envelope)
        events = detect_events(envelope, self.settings.events)
        hypnogram = load_hypnogram(segment.hypnogram_path) if segment.hypnogram_path else None
        spo2, desaturations = self.desaturations()
        if desaturations is not None:
            events["desaturation"] = event_desaturation(events, desaturations, self.settings.oximetry.lag_seconds)
        events.to_csv(self.paths.events, index=False)

        frame = self.extract_features(signal, baseline if cfg.normalisation == "recording" else None)
        starts, ends = frame[META_COLUMNS[0]].to_numpy(float), frame[META_COLUMNS[1]].to_numpy(float)
        frame["event"] = window_event_labels(starts, ends, events, self.settings.events.min_overlap_seconds)
        if hypnogram is not None:
            frame["stage"] = window_stages(starts, ends, hypnogram)
        summary = event_summary(self.stage, events, frame["event"].to_numpy(object), None, signal.duration, baseline, hypnogram)
        if desaturations is not None:
            summary = summary.model_copy(update=oximetry_summary(events, desaturations, summary.sleep_hours, hypnogram, self.settings.oximetry))
            self.log.info("oximetry: desaturations=%d odi=%s confirmed events=%s ahi (confirmed hypopneas)=%s", summary.desaturations,
                          _fmt(summary.odi), _fmt(summary.confirmed_fraction), _fmt(summary.ahi_confirmed))
        self.paths.event_summary.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        self.log.info("events: apnea=%d hypopnea=%d artefact=%d events/hour=%.1f ahi=%s event windows=%.1f%% (baseline amplitude %.3g)",
                      summary.apnea, summary.hypopnea, summary.artefact, summary.events_per_hour,
                      None if summary.ahi is None else f"{summary.ahi:.1f}", 100 * summary.event_window_fraction, baseline)
        self.hmm_detector(envelope, desaturations, hypnogram, frame, signal.duration, baseline)
        night = "Night 1 (training)" if self.stage == "training" else "Night 2 (validation)"
        save_events_overview(envelope, events, hypnogram, baseline, f"{night} at a glance: breathing, events, oxygen and sleep", self.paths.plots / "events.png", spo2=spo2,
                             baselines=baseline_variants(envelope["envelope"].to_numpy(float), self.settings.events))

        features = feature_matrix(frame)
        X = self.transformer(features).transform(features)
        self.log.info("clustering %d windows x %d columns %s", *X.shape, list(X.columns))
        return Prepared(signal, envelope, baseline, events, hypnogram, frame, X)

    def desaturations(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """1 Hz SpO2 and its desaturations (also written to desaturations.csv); (None, None) without an oximetry channel."""
        cfg, segment = self.settings.oximetry, self.settings.segment(self.stage)
        if cfg.channel is None:
            return None, None
        try:
            spo2 = load_channel(segment.file_path, cfg.channel, segment.start_seconds, segment.end_seconds)
        except ValueError as error:
            self.log.warning("no oximetry reference: %s", error)
            return None, None
        series = spo2_series(spo2, cfg)
        desaturations = detect_desaturations(series, cfg)
        desaturations.to_csv(self.paths.desaturations, index=False, columns=DESATURATION_COLUMNS)
        return series, desaturations

    def hmm_detector(self, envelope: pd.DataFrame, desaturations: pd.DataFrame | None, hypnogram: pd.DataFrame | None,
                     frame: pd.DataFrame, duration: float, baseline: float) -> pd.DataFrame:
        """HMM events of the stage (hmm_events.csv / .json); the model is fitted on training and only applied to validation."""
        path, key = self.paths.models / "hmm.pkl", {"hmm": self.settings.hmm.model_dump(), "events": self.settings.events.model_dump()}
        model = GaussianHMM.load(path) if path.exists() else None
        if model is None and self.stage == "validation":
            raise FileNotFoundError(f"no trained HMM {path}; run training first")
        if self.stage == "training" and (model is None or not self.resume or getattr(model, "settings_", None) != key):
            t0 = time.perf_counter()
            model = fit_hmm(envelope["envelope"].to_numpy(float), self.settings.events, self.settings.hmm)
            model.settings_ = key
            model.save(path)
            self.log.info("hmm: fitted %d states in %d iterations (%.1fs): mean flow %s, stay %s, names %s", model.n_states, model.n_iter_,
                          time.perf_counter() - t0, np.round(model.mean_flow, 2).tolist(), np.round(np.diag(model.transitions_), 3).tolist(), model.names_)
        events, _ = hmm_events(envelope, model, self.settings.events, self.settings.hmm)
        if desaturations is not None:
            events["desaturation"] = event_desaturation(events, desaturations, self.settings.oximetry.lag_seconds)
        events.to_csv(self.paths.hmm_events, index=False)
        starts, ends = frame[META_COLUMNS[0]].to_numpy(float), frame[META_COLUMNS[1]].to_numpy(float)
        labels = window_event_labels(starts, ends, events, self.settings.events.min_overlap_seconds)
        summary = event_summary(self.stage, events, labels, None, duration, baseline, hypnogram)
        if desaturations is not None:
            summary = summary.model_copy(update=oximetry_summary(events, desaturations, summary.sleep_hours, hypnogram, self.settings.oximetry))
        self.paths.hmm_summary.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        self.log.info("hmm events: apnea=%d hypopnea=%d artefact=%d ahi=%s confirmed=%s", summary.apnea, summary.hypopnea, summary.artefact,
                      _fmt(summary.ahi), _fmt(summary.confirmed_fraction))
        return events

    # ----------------------------------------------------------------- features
    def cache_path(self) -> Path:
        return self.paths.cache / f"features_{self.stage}_{self.settings.feature_key(self.stage)}.csv"

    def extract_features(self, signal: SignalData, scale: float | None = None) -> pd.DataFrame:
        """Window features, appended to a CSV cache in chunks so an interrupted run resumes.

        `scale` is the recording baseline amplitude for `normalisation: recording`; None scales each window by its own std.
        """
        cfg = self.settings.data
        window = int(round(cfg.window_seconds * signal.sfreq))
        starts = window_starts(len(signal.signal), window, int(round(cfg.hop_seconds * signal.sfreq)))
        cache = self.cache_path()
        if cache.exists() and not self.resume:
            cache.unlink()
        done = _valid_cached_rows(cache, signal.start_time + starts / signal.sfreq)
        if done:
            self.log.info("resuming feature extraction: %d/%d windows cached in %s", done, len(starts), cache.name)

        if done < len(starts):
            preprocess = WindowPreprocessor(signal.sfreq, cfg.smoothing_seconds, scale=scale)
            extractor = make_extractor(self.settings.features.extractor, signal.sfreq)
            self.log.info("extracting %s features for %d windows (%s scaling%s)", self.settings.features.extractor, len(starts) - done,
                          cfg.normalisation, "" if scale is None else f", baseline {scale:.3g}")
            t0 = time.perf_counter()
            with Parallel(n_jobs=self.settings.run.n_jobs) as parallel:
                for i in range(done, len(starts), self.settings.run.chunk_windows):
                    chunk = starts[i : i + self.settings.run.chunk_windows]
                    rows = parallel(delayed(window_features)(signal.signal[s : s + window], preprocess, extractor) for s in chunk)
                    out = pd.DataFrame(rows)
                    out.insert(0, META_COLUMNS[1], signal.start_time + (chunk + window) / signal.sfreq)
                    out.insert(0, META_COLUMNS[0], signal.start_time + chunk / signal.sfreq)
                    _append_csv(out, cache)
                    self.log.info("features %d/%d windows (%.1fs)", i + len(chunk), len(starts), time.perf_counter() - t0)

        return self._clean(pd.read_csv(cache))

    def _clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = [c for c in frame.columns if c not in META_COLUMNS]
        values = frame[columns].replace([np.inf, -np.inf], np.nan)
        bad = int(values.isna().sum().sum())
        if bad:
            self.log.warning("%d non-finite feature values replaced by column medians", bad)
            frame[columns] = values.fillna(values.median())
        return frame

    # -------------------------------------------------------------- transformer
    def transformer(self, features: pd.DataFrame) -> FeatureTransformer:
        path = self.paths.models / "feature_transformer.pkl"
        if path.exists():
            transformer = FeatureTransformer.load(path)
            if transformer.cfg == self.settings.features and (self.stage == "validation" or self.resume):
                self.log.info("loaded feature transformer from %s (columns=%s)", path, transformer.output_columns)
                return transformer
        if self.stage == "validation":
            raise FileNotFoundError(f"no feature transformer matching current settings in {path}; run training first")
        transformer = FeatureTransformer(self.settings.features).fit(features)
        transformer.save(path)
        self.log.info("fitted feature transformer: selected=%s output=%s", transformer.columns_, transformer.output_columns)
        return transformer

    # ----------------------------------------------------------------- clusters
    def cluster(self, name: str, params: dict, prepared: Prepared) -> ClusterMetrics:
        path = self.paths.models / f"{name}.pkl"
        values = prepared.X.to_numpy(float)
        model, fit_seconds, resumed = self._model(name, params, values, path)
        labels = model.predict(values)
        metrics = evaluate(values, labels, algorithm=name, stage=self.stage, fit_seconds=fit_seconds, resumed=resumed,
                           event_labels=prepared.event_labels, stages=prepared.stages)

        profile = cluster_profile(labels, prepared.event_labels, prepared.stages, feature_matrix(prepared.frame))
        names = self._names(model, profile, path)
        profile.insert(1, "name", profile["cluster_id"].map(names))
        metrics.cluster_names = {str(k): v for k, v in names.items()}

        out = prepared.frame.copy()
        out["cluster_id"] = labels
        out["cluster_label"] = out["cluster_id"].map(names)
        out.to_csv(self.paths.csvs / f"{name}.csv", index=False)
        profile.to_csv(self.paths.csvs / f"{name}_clusters.csv", index=False)
        night = "night 1" if self.stage == "training" else "night 2"
        title = f"{name} on {prepared.signal.channel} ({self.stage})"
        save_scatter(values, labels, model.centers_, f"{name}: the groups on a map of all 30 s windows ({night})", self.paths.plots / f"{name}.png", names)
        save_timeline(out, prepared.events, f"{name}: which group every 30 s of {night} falls in", self.paths.plots / f"{name}_timeline.png", names)
        if self.settings.run.interactive_plots:
            save_interactive(prepared.signal, out, title, self.paths.plots / f"{name}.html", self.settings.run.plot_max_points,
                             events=prepared.events, hypnogram=prepared.hypnogram, names=names)
        self.log.info("%s: clusters=%d noise=%d silhouette=%s quality=%s event_f1=%s event_clusters=%s names=%s", name, metrics.n_clusters,
                      metrics.n_noise, _fmt(metrics.silhouette), metrics.quality, _fmt(metrics.event_f1), metrics.event_clusters, names)
        return metrics

    def _names(self, model: Clusterer, profile: pd.DataFrame, path: Path) -> dict[int, str]:
        """Cluster names are decided on training data and stored with the model so validation reuses them."""
        names = dict(getattr(model, "names_", {}) or {})
        if self.stage == "training" and not names:
            names = name_clusters(profile)
            model.names_ = names
            model.save(path)
        for k in profile["cluster_id"].astype(int):
            names.setdefault(k, "noise" if k < 0 else f"cluster {k}")  # labels unseen in training (e.g. noise at validation)
        return names

    def _model(self, name: str, params: dict, X: np.ndarray, path: Path) -> tuple[Clusterer, float, bool]:
        if path.exists():
            model = Clusterer.load(path)
            if self.stage == "validation":
                return model, 0.0, True
            if self.resume and model.params == params:
                self.log.info("%s: resumed fitted model from %s", name, path)
                return model, 0.0, True
        elif self.stage == "validation":
            raise FileNotFoundError(f"no trained model {path}; run training first")
        t0 = time.perf_counter()
        model = make_clusterer(name, params).fit(X)
        model.save(path)
        return model, time.perf_counter() - t0, False


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _valid_cached_rows(path: Path, expected_starts: np.ndarray) -> int:
    """Keep the longest cache prefix whose rows are complete and match the expected windows."""
    if not path.exists():
        return 0
    cached = pd.read_csv(path, on_bad_lines="skip")
    n = min(len(cached), len(expected_starts))
    head = cached.iloc[:n]
    ok = np.isclose(head[META_COLUMNS[0]].to_numpy(float), expected_starts[:n]) & head.iloc[:, -1].notna().to_numpy()
    valid = n if ok.all() else int(np.argmin(ok))
    if valid != len(cached):
        cached.iloc[:valid].to_csv(path, index=False)
    return valid


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    exists = path.exists() and path.stat().st_size > 0
    frame.to_csv(path, mode="a", header=not exists, index=False)
