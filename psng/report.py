"""Cross-stage, cross-algorithm comparison report and the number-of-clusters sweep."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from psng.clustering import make_clusterer
from psng.data import EventSummary
from psng.events import baseline_variants, detect_events, event_summary, flow_envelope, load_hypnogram, window_event_labels
from psng.examples import Recording, example_figures, load_recordings
from psng.metrics import evaluate, read_metrics
from psng.data import feature_matrix
from psng.pipeline import Pipeline
from psng.transform import FeatureTransformer
from psng.plots import (COMPARISON_METRICS, save_comparison, save_embedding, save_event_rates, save_importances, save_kalman_excerpt,
                        save_latent_spaces, save_spaces_heatmap, save_sweep)
from psng.settings import ReductionSettings, Settings
from psng.supervised import run_supervised

STAGES = ("training", "validation")
SPACES = {  # feature spaces compared by `python -m psng compare`
    "pca3": {"type": "pca", "n_components": 3},
    "raw": {"type": "none"},
    "autoencoder3": {"type": "autoencoder", "n_components": 3},
}
K_PARAM = {"kmeans": "n_clusters", "minibatch": "n_clusters", "bisecting": "n_clusters", "gmm": "n_components"}


def build_report(output_dir: Path, docs_dir: Path | None = None, settings: Settings | None = None) -> str:
    """Read both stages' outputs, write comparison figures + report.md and return the markdown.

    With `settings` the recordings are re-read to draw example time slices per event kind and per cluster.
    """
    output_dir = Path(output_dir)
    metrics = pd.concat([pd.DataFrame([m.model_dump() for m in read_metrics(output_dir / s / "metrics.json")])
                         for s in STAGES if (output_dir / s / "metrics.json").exists()], ignore_index=True)
    if metrics.empty:
        raise FileNotFoundError(f"no metrics.json under {output_dir}; run training (and validation) first")
    summaries = [EventSummary.model_validate(json.loads((output_dir / s / "events.json").read_text(encoding="utf-8")))
                 for s in STAGES if (output_dir / s / "events.json").exists()]
    profiles = _profiles(output_dir, metrics)

    comparison = output_dir / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    save_comparison(metrics, comparison / "metrics.png")
    if not profiles.empty:
        save_event_rates(profiles, comparison / "event_rates.png")
    models = list(dict.fromkeys(metrics["algorithm"]))
    baselines = pd.DataFrame()
    if settings is not None:
        recordings = load_recordings(settings)
        example_figures(settings, models, recordings=recordings)
        embedding_figure(output_dir, models[0])
        baselines = baseline_comparison(settings, recordings)
        latent_figure(output_dir)

    lines = ["# PSNG performance report", ""]
    if summaries:
        lines += ["## Rule-based airflow events per recording", "",
                  _markdown_table(pd.DataFrame([s.model_dump() for s in summaries]).drop(columns=["n_windows"]), {
                      "duration_hours": "{:.2f}", "sleep_hours": "{:.2f}", "events_per_hour": "{:.1f}", "ahi": "{:.1f}",
                      "event_window_fraction": "{:.1%}", "baseline_amplitude": "{:.3g}"}), ""]
    lines += ["## Clustering metrics per algorithm and stage", "",
              _markdown_table(metrics[[c for c in ("algorithm", "stage", "n_clusters", "n_noise", "silhouette", "davies_bouldin",
                                                   "calinski_harabasz", "event_f1", "event_precision", "event_recall", "event_nmi",
                                                   "stage_nmi", "fit_seconds") if c in metrics.columns]],
                              {c: "{:.3f}" for c in ("silhouette", "davies_bouldin", "event_f1", "event_precision", "event_recall",
                                                     "event_nmi", "stage_nmi", "fit_seconds")} | {"calinski_harabasz": "{:.1f}"}), ""]
    if not baselines.empty:
        lines += ["## Event detection: rolling-median vs Kalman baseline", "",
                  _markdown_table(baselines, {"events_per_hour": "{:.1f}", "ahi": "{:.1f}", "event_window_fraction": "{:.1%}"}, highlight=set()), ""]
    best = _best(metrics)
    if best:
        lines += ["## Best algorithm per metric (validation)", ""] + [f"* **{label}**: {algo} ({value:.3f})" for label, algo, value in best] + [""]
    lines += ["## Figures", "", "* `comparison/metrics.png` - every metric per algorithm, training next to validation",
              "* `comparison/event_rates.png` - apnea / hypopnea share per cluster",
              "* `<stage>/plots/events.png` - airflow envelope with detected events and the hypnogram",
              "* `<stage>/plots/<model>_timeline.png` - cluster of every window over the night, events shaded",
              "* `comparison/event_examples.png` - example signal slices per event kind, training next to validation",
              "* `comparison/<model>_examples.png` - example windows per cluster, training next to validation", ""]
    markdown = "\n".join(lines)
    (comparison / "report.md").write_text(markdown, encoding="utf-8")
    if docs_dir is not None:
        copy_figures(output_dir, Path(docs_dir), models)
    return markdown


def copy_figures(output_dir: Path, docs_dir: Path, models: list[str]) -> list[Path]:
    """Copy the comparison, event and timeline PNGs into `docs_dir/figures` so a README can show them."""
    figures = docs_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    copied = []
    sources = [output_dir / "comparison" / name for name in ("metrics.png", "event_rates.png", "event_examples.png", "spaces.png", "feature_importance.png",
                                                             "tsne.png", "kalman.png", "xgboost.png", "autoencoder.png")]
    sources += [output_dir / "training" / f"sweep_{m}.png" for m in K_PARAM]
    sources += [output_dir / "comparison" / f"{m}_examples.png" for m in models]
    for stage in STAGES:
        sources.append(output_dir / stage / "plots" / "events.png")
        sources += [output_dir / stage / "plots" / f"{m}_timeline.png" for m in models]
    for src in sources:
        if src.exists():
            dst = figures / (f"{src.parent.parent.name}_{src.name}" if src.parent.name == "plots" else src.name)
            shutil.copyfile(src, dst)
            copied.append(dst)
    return copied


def sweep(settings: Settings, model: str, ks: list[int]) -> pd.DataFrame:
    """Fit `model` with every k on the training features and score each; writes sweep_<model>.csv / .png under training."""
    if model not in K_PARAM:
        raise ValueError(f"sweep supports {sorted(K_PARAM)}, not {model!r}")
    pipeline = Pipeline(settings, "training")
    prepared = pipeline.prepare()
    X = prepared.X.to_numpy(float)
    rows = []
    for k in ks:
        clusterer = make_clusterer(model, {K_PARAM[model]: int(k)}).fit(X)
        m = evaluate(X, clusterer.predict(X), algorithm=model, stage="training", event_labels=prepared.event_labels, stages=prepared.stages)
        rows.append({"k": int(k), **m.model_dump(include={"silhouette", "davies_bouldin", "calinski_harabasz", "event_f1", "event_nmi", "stage_nmi"})})
        pipeline.log.info("sweep %s k=%d silhouette=%s event_f1=%s", model, k, _f(m.silhouette), _f(m.event_f1))
    results = pd.DataFrame(rows)
    results.to_csv(pipeline.paths.stage_dir / f"sweep_{model}.csv", index=False)
    save_sweep(results, model, pipeline.paths.stage_dir / f"sweep_{model}.png")
    return results


# ------------------------------------------------------------------ helpers
def _profiles(output_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for stage in STAGES:
        for algorithm in metrics.loc[metrics["stage"] == stage, "algorithm"]:
            path = output_dir / stage / "csvs" / f"{algorithm}_clusters.csv"
            if path.exists():
                frames.append(pd.read_csv(path).assign(algorithm=algorithm, stage=stage))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _best(metrics: pd.DataFrame) -> list[tuple[str, str, float]]:
    rows = metrics[metrics["stage"] == "validation"] if "validation" in set(metrics["stage"]) else metrics
    out = []
    for metric, label in COMPARISON_METRICS:
        if metric not in rows.columns or rows[metric].isna().all():
            continue
        idx = rows[metric].idxmin() if metric == "davies_bouldin" else rows[metric].idxmax()
        out.append((label, str(rows.loc[idx, "algorithm"]), float(rows.loc[idx, metric])))
    return out


LOWER_IS_BETTER = {"davies_bouldin", "n_noise"}
SCORE_COLUMNS = {"silhouette", "davies_bouldin", "calinski_harabasz", "event_f1", "event_precision", "event_recall", "event_nmi", "stage_nmi",
                 "validation_f1", "validation_auc"}


def _markdown_table(frame: pd.DataFrame, formats: dict[str, str], highlight: set[str] | None = None, group: str | None = "stage") -> str:
    """Markdown table; the best value of every `highlight` column (per `group`, e.g. stage) is bold."""
    highlight = {c for c in (SCORE_COLUMNS if highlight is None else highlight) if c in frame.columns}
    best: set[tuple[int, str]] = set()
    for _, part in (frame.groupby(group) if group in frame.columns else [(None, frame)]):
        for column in highlight:
            values = pd.to_numeric(part[column], errors="coerce")
            if values.notna().sum() > 1:
                best.add((values.idxmin() if column in LOWER_IS_BETTER else values.idxmax(), column))

    def cell(index, column: str, value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        text = formats.get(column, "{}").format(value)
        return f"**{text}**" if (index, column) in best else text

    header = "| " + " | ".join(frame.columns) + " |"
    sep = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(i, c, v) for c, v in row.items()) + " |" for i, row in frame.iterrows()]
    return "\n".join([header, sep, *body])


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# ------------------------------------------------------------------ baseline trackers
def baseline_comparison(settings: Settings, recordings: dict[str, Recording]) -> pd.DataFrame:
    """Events found per night with the rolling-median and the Kalman baseline (same thresholds); writes comparison/baselines.csv."""
    rows = []
    for stage, rec in recordings.items():
        envelope = flow_envelope(rec.signal, settings.data.smoothing_seconds)
        segment = settings.segment(stage)
        hypnogram = load_hypnogram(segment.hypnogram_path) if segment.hypnogram_path else None
        starts = np.arange(rec.signal.start_time, rec.signal.start_time + rec.signal.duration - settings.data.window_seconds + 1e-9, settings.data.window_seconds)
        for method in ("rolling_median", "kalman"):
            cfg = settings.events.model_copy(update={"baseline": method})
            events = detect_events(envelope, cfg)
            labels = window_event_labels(starts, starts + settings.data.window_seconds, events, cfg.min_overlap_seconds)
            summary = event_summary(stage, events, labels, None, rec.signal.duration, rec.baseline, hypnogram)
            rows.append({"stage": stage, "baseline": method, **summary.model_dump(include={"apnea", "hypopnea", "artefact", "events_per_hour", "ahi", "event_window_fraction"})})
    table = pd.DataFrame(rows)
    table.to_csv(settings.run.output_dir / "comparison" / "baselines.csv", index=False)
    kalman_figure(settings, recordings)
    return table


def kalman_figure(settings: Settings, recordings: dict[str, Recording], seconds: float = 3600.0) -> Path | None:
    """The busiest hour of the last recording: envelope, both baseline trackers and the events each one finds."""
    if not recordings:
        return None
    stage, rec = list(recordings.items())[-1]
    envelope = flow_envelope(rec.signal, settings.data.smoothing_seconds)
    env = envelope["envelope"].to_numpy(float)
    baselines = baseline_variants(env, settings.events)
    events = {m: detect_events(envelope, settings.events.model_copy(update={"baseline": m})) for m in baselines}
    reference = events[settings.events.baseline]
    starts = np.arange(rec.signal.start_time, rec.signal.start_time + rec.signal.duration - seconds + 1, 600.0)
    busiest = max(starts, key=lambda t: ((reference["start_time"] >= t) & (reference["start_time"] < t + seconds)).sum()) if len(starts) else rec.signal.start_time
    path = settings.run.output_dir / "comparison" / "kalman.png"
    save_kalman_excerpt(envelope, baselines, events, float(np.median(env)), float(busiest), seconds,
                        f"Kalman vs rolling-median baseline ({stage}, busiest hour)", path)
    return path


def latent_figure(output_dir: Path, spaces: dict[str, Path] | None = None) -> Path | None:
    """First two components of the PCA and autoencoder spaces per stage, coloured by event kind (needs `compare` outputs)."""
    spaces = spaces or {"PCA": output_dir / "training" / "models" / "feature_transformer.pkl",
                        "autoencoder": output_dir / "spaces" / "autoencoder3" / "training" / "models" / "feature_transformer.pkl"}
    frames = {s: pd.read_csv(next((output_dir / s / "csvs").glob("*.csv"))) for s in STAGES if list((output_dir / s / "csvs").glob("*.csv"))}
    spaces = {name: FeatureTransformer.load(p) for name, p in spaces.items() if p.exists()}
    if len(spaces) < 2 or not frames:
        return None
    data = {name: {s: (t.transform(feature_matrix(f.drop(columns=["cluster_id", "cluster_label"], errors="ignore"))), f["event"].to_numpy(object))
                   for s, f in frames.items()} for name, t in spaces.items()}
    path = output_dir / "comparison" / "autoencoder.png"
    save_latent_spaces(data, "Reduced feature spaces coloured by rule-based event kind", path)
    return path


# ------------------------------------------------------------------ t-SNE (visualisation only)
def embedding_figure(output_dir: Path, model: str, perplexity: float = 30.0) -> Path | None:
    """t-SNE of each stage's clustered windows, coloured by `model` cluster. Visualisation only: t-SNE cannot
    project unseen windows, so each stage is embedded on its own and never used for clustering."""
    from sklearn.manifold import TSNE

    frames = {s: pd.read_csv(output_dir / s / "csvs" / f"{model}.csv") for s in STAGES if (output_dir / s / "csvs" / f"{model}.csv").exists()}
    if not frames:
        return None
    transformer = FeatureTransformer.load(output_dir / "training" / "models" / "feature_transformer.pkl")
    points = {s: TSNE(n_components=2, perplexity=min(perplexity, len(f) - 1), random_state=0, init="pca").fit_transform(
        transformer.transform(feature_matrix(f.drop(columns=["cluster_id", "cluster_label"]))).to_numpy(float)) for s, f in frames.items()}
    profile = output_dir / "training" / "csvs" / f"{model}_clusters.csv"
    names = dict(zip(*(pd.read_csv(profile)[c] for c in ("cluster_id", "name")))) if profile.exists() else {}
    (output_dir / "comparison").mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison" / "tsne.png"
    save_embedding(points, frames, {int(k): str(v) for k, v in names.items()}, f"t-SNE of the clustered feature space ({model} clusters)", path)
    return path


# ------------------------------------------------------------------ side-by-side feature spaces + supervised reference
def compare_spaces(settings: Settings, spaces: dict[str, dict] = SPACES) -> tuple[pd.DataFrame, str]:
    """Train + validate every model in every feature space; returns the metrics table and a markdown summary.

    Each space gets its own output folder (`<output_dir>/spaces/<name>`) and reuses the main feature cache. Writes
    comparison/spaces.{csv,png,md}, comparison/supervised.json and comparison/feature_importance.png.
    """
    base = settings.run.output_dir
    rows = []
    for name, reduction in spaces.items():
        s = settings.model_copy(deep=True)
        s.features.reduction = ReductionSettings(**reduction)
        s.run.output_dir, s.run.log_dir, s.run.interactive_plots = base / "spaces" / name, settings.run.log_dir / "spaces" / name, False
        (s.run.output_dir / "cache").mkdir(parents=True, exist_ok=True)
        for cache in (base / "cache").glob("features_*.csv"):
            if not (s.run.output_dir / "cache" / cache.name).exists():
                shutil.copyfile(cache, s.run.output_dir / "cache" / cache.name)
        for stage in STAGES:
            rows += [m.model_dump() | {"space": name} for m in Pipeline(s, stage).run()]
    results = pd.DataFrame(rows)
    comparison = base / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    results.to_csv(comparison / "spaces.csv", index=False)

    supervised = run_supervised(base, f"{next(iter(settings.models))}.csv")
    reference = None
    if supervised is not None:
        reference = {"event_f1": supervised.validation_f1}
        if supervised.importances:
            save_importances(supervised.importances, f"{supervised.backend}: feature importance for event windows (trained on night 1)", comparison / "feature_importance.png")
    save_spaces_heatmap(results, comparison / "spaces.png", reference=reference)

    columns = ["space", "algorithm", "stage", "n_clusters", "n_noise", "silhouette", "davies_bouldin", "event_f1", "event_precision", "event_recall", "event_nmi", "stage_nmi"]
    lines = ["## Feature spaces side by side", "", _markdown_table(results[columns].sort_values(["stage", "algorithm", "space"]), {c: "{:.3f}" for c in columns[5:]}), ""]
    pivot = results[results["stage"] == "validation"].pivot_table(index="algorithm", columns="space", values="event_f1", aggfunc="first").reset_index()
    lines += ["### Validation event F1 per algorithm and feature space", "",
              _markdown_table(pivot, {c: "{:.3f}" for c in pivot.columns[1:]}, highlight=set(pivot.columns[1:]), group=None), ""]
    if supervised is not None:
        lines += ["## Supervised reference (gradient boosting on the rule-based event labels)", "",
                  _markdown_table(pd.DataFrame([supervised.model_dump(exclude={"importances"})]),
                                  {c: "{:.3f}" for c in ("train_f1", "validation_f1", "validation_precision", "validation_recall", "validation_auc")}), "",
                  "Top features: " + ", ".join(f"{k} ({v:.2f})" for k, v in list(supervised.importances.items())[:6]), ""]
    markdown = "\n".join(lines)
    (comparison / "spaces.md").write_text(markdown, encoding="utf-8")
    return results, markdown
