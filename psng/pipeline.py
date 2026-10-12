"""Training / validation orchestration with resumable checkpoints.

Training:   EDF -> windows -> features (cached per chunk) -> fit transformer -> fit models -> outputs
Validation: EDF -> windows -> features (cached per chunk) -> load transformer -> load models -> outputs
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from psng.clustering import Clusterer, make_clusterer
from psng.data import META_COLUMNS, ClusterMetrics, SignalData, feature_matrix
from psng.edf import load_channel
from psng.features import FeatureExtractor, make_extractor
from psng.log import get_logger
from psng.metrics import evaluate, format_table, write_metrics
from psng.plots import save_interactive, save_scatter
from psng.settings import Settings, Stage
from psng.transform import FeatureTransformer
from psng.windows import WindowPreprocessor, window_starts

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # loky cannot count physical cores on Windows


def window_features(window: np.ndarray, preprocess: WindowPreprocessor, extractor: FeatureExtractor) -> dict[str, float]:
    return extractor.extract(preprocess(window))


class Paths:
    def __init__(self, output_dir: Path, stage: Stage):
        self.models = output_dir / "training" / "models"  # models are always read from training
        self.cache = output_dir / "cache"
        self.csvs = output_dir / stage / "csvs"
        self.plots = output_dir / stage / "plots"
        self.metrics = output_dir / stage / "metrics.json"
        for folder in (self.models, self.cache, self.csvs, self.plots):
            folder.mkdir(parents=True, exist_ok=True)


class Pipeline:
    def __init__(self, settings: Settings, stage: Stage):
        self.settings = settings
        self.stage = stage
        self.resume = settings.run.resume
        self.paths = Paths(settings.run.output_dir, stage)
        self.log = get_logger(f"psng.{stage}", settings.run.log_dir / f"{stage}.log", settings.run.log_level)

    def run(self) -> list[ClusterMetrics]:
        segment = self.settings.segment(self.stage)
        self.log.info("stage=%s file=%s channel=%s segment=%s-%ss resume=%s",
                      self.stage, segment.file_path, self.settings.data.channel, segment.start_seconds, segment.end_seconds, self.resume)
        signal = load_channel(segment.file_path, self.settings.data.channel, segment.start_seconds, segment.end_seconds)
        self.log.info("loaded %.1f s at %g Hz", signal.duration, signal.sfreq)

        frame = self.extract_features(signal)
        features = feature_matrix(frame)
        transformer = self.transformer(features)
        X = transformer.transform(features)
        self.log.info("clustering %d windows x %d columns %s", *X.shape, list(X.columns))

        metrics = [self.cluster(name, params, X, frame, signal) for name, params in self.settings.models.items()]
        write_metrics(metrics, self.paths.metrics)
        self.log.info("metrics written to %s\n%s", self.paths.metrics, format_table(metrics))
        return metrics

    # ----------------------------------------------------------------- features
    def cache_path(self) -> Path:
        return self.paths.cache / f"features_{self.stage}_{self.settings.feature_key(self.stage)}.csv"

    def extract_features(self, signal: SignalData) -> pd.DataFrame:
        """Window features, appended to a CSV cache in chunks so an interrupted run resumes."""
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
            preprocess = WindowPreprocessor(signal.sfreq, cfg.smoothing_seconds)
            extractor = make_extractor(self.settings.features.extractor, signal.sfreq)
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
    def cluster(self, name: str, params: dict, X: pd.DataFrame, frame: pd.DataFrame, signal: SignalData) -> ClusterMetrics:
        path = self.paths.models / f"{name}.pkl"
        values = X.to_numpy(float)
        model, fit_seconds, resumed = self._model(name, params, values, path)
        labels = model.predict(values)
        metrics = evaluate(values, labels, algorithm=name, stage=self.stage, fit_seconds=fit_seconds, resumed=resumed)

        out = frame.copy()
        out["cluster_id"] = labels
        out["cluster_label"] = "cluster_label " + out["cluster_id"].astype(str)
        out.to_csv(self.paths.csvs / f"{name}.csv", index=False)
        title = f"{name} on {signal.channel} ({self.stage})"
        save_scatter(values, labels, model.centers_, title, self.paths.plots / f"{name}.png")
        if self.settings.run.interactive_plots:
            save_interactive(signal, out, title, self.paths.plots / f"{name}.html", self.settings.run.plot_max_points)
        self.log.info("%s: clusters=%d noise=%d silhouette=%s quality=%s", name, metrics.n_clusters, metrics.n_noise,
                      None if metrics.silhouette is None else f"{metrics.silhouette:.3f}", metrics.quality)
        return metrics

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
