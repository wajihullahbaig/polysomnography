"""Cross-stage, cross-algorithm comparison report and the number-of-clusters sweep."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from psng.cards import build_cards
from psng.clustering import make_clusterer
from psng.data import EventSummary
from psng.events import (baseline_variants, detect_events, event_summary, flow_envelope, load_hypnogram,
                         window_event_labels)
from psng.examples import Recording, example_figures, load_recordings
from psng.metrics import evaluate, read_metrics
from psng.oximetry import desaturation_scores, load_spo2, window_intervals
from psng.data import feature_matrix
from psng.pipeline import Pipeline
from psng.transform import FeatureTransformer
from psng.plots import (COMPARISON_METRICS, save_comparison, save_embedding, save_event_rates, save_importances, save_kalman_excerpt,
                        save_latent_spaces, save_oximetry_excerpt, save_oximetry_scores, save_spaces_heatmap, save_sweep)
from psng.settings import ReductionSettings, Settings
from psng.supervised import event_classifier, run_supervised

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
    summaries = [EventSummary.model_validate(json.loads((output_dir / s / name).read_text(encoding="utf-8"))).model_dump() | {"detector": detector}
                 for detector, name in (("rules", "events.json"), ("HMM", "hmm_events.json")) for s in STAGES if (output_dir / s / name).exists()]
    profiles = _profiles(output_dir, metrics)

    comparison = output_dir / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    save_comparison(metrics, comparison / "metrics.png")
    if not profiles.empty:
        save_event_rates(profiles, comparison / "event_rates.png")
    models = list(dict.fromkeys(metrics["algorithm"]))
    baselines = oximetry = agreement = pd.DataFrame()
    if settings is not None:
        recordings = load_recordings(settings)
        example_figures(settings, models, recordings=recordings)
        embedding_figure(output_dir, models[0])
        baselines = baseline_comparison(settings, recordings)
        latent_figure(output_dir)
        oximetry = oximetry_comparison(settings, recordings, models)
        agreement = hmm_agreement(output_dir, models[0])
        build_cards(settings, recordings, models)

    lines = ["# PSNG performance report", ""]
    if summaries:
        table = pd.DataFrame(summaries).drop(columns=["n_windows"])
        lines += ["## Airflow events per recording: threshold rules and HMM", "",
                  _markdown_table(table[["detector", *table.columns.drop("detector")]], {
                      "duration_hours": "{:.2f}", "sleep_hours": "{:.2f}", "events_per_hour": "{:.1f}", "ahi": "{:.1f}",
                      "event_window_fraction": "{:.1%}", "baseline_amplitude": "{:.3g}", "odi": "{:.1f}", "confirmed_fraction": "{:.1%}", "ahi_confirmed": "{:.1f}"}), ""]
    if not oximetry.empty:
        lines += ["## Every detector against SpO2 desaturations (sleep only)", "",
                  _markdown_table(oximetry, {c: "{:.3f}" for c in ("confirmed", "chance", "explained", "f1")} | {"lift": "{:.2f}", "events_per_hour": "{:.1f}"},
                                  highlight={"confirmed", "lift", "explained", "f1"}), ""]
    if not agreement.empty:
        lines += ["## HMM events against the rule-based event windows", "",
                  _markdown_table(agreement, {c: "{:.3f}" for c in ("event_f1", "event_precision", "event_recall")}, highlight=set()), ""]
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
              "* `comparison/oximetry.png` / `oximetry_scores.png` - events against SpO2: an excerpt, and every detector scored",
              "* `comparison/<algorithm>_card.png` - one card per algorithm (rules, clusterers, HMM, XGBoost) in the same layout",
              "* `comparison/<algorithm>_examples.png` - example signal slices per group, or where the algorithm agrees / disagrees with the rules",
              "* `comparison/event_rates.png` - apnea / hypopnea share per cluster",
              "* `<stage>/plots/events.png` - airflow envelope with detected events and the hypnogram",
              "* `<stage>/plots/<model>_timeline.png` - cluster of every window over the night, events shaded",
                            ""]
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
    sources = [output_dir / "comparison" / name for name in ("metrics.png", "event_rates.png", "spaces.png", "feature_importance.png",
                                                             "tsne.png", "kalman.png", "xgboost.png", "autoencoder.png",
                                                             "oximetry.png", "oximetry_scores.png")]
    sources += [output_dir / "training" / f"sweep_{m}.png" for m in K_PARAM]
    sources += sorted((output_dir / "comparison").glob("*_card.png")) + sorted((output_dir / "comparison").glob("*_examples.png"))
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
    save_latent_spaces(data, "Compressed feature maps: PCA (straight lines) vs autoencoder (curved)", path)
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
    save_embedding(points, frames, {int(k): str(v) for k, v in names.items()}, f"A map of every 30 s window (t-SNE), coloured by {model} group", path)
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
            save_importances(supervised.importances, "What XGBoost looks at to spot an event window", comparison / "feature_importance.png")
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


# ------------------------------------------------------------------ SpO2 reference
def detector_intervals(settings: Settings, rec: Recording, models: list[str], stage: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Events of every detector as (starts, ends): both rule baselines, each clusterer's event clusters, and XGBoost on validation.

    Event clusters are the ones chosen on the training night, so validation is scored without looking at its labels.
    """
    envelope = flow_envelope(rec.signal, settings.data.smoothing_seconds)
    out = {}
    for method in ("rolling_median", "kalman"):
        events = detect_events(envelope, settings.events.model_copy(update={"baseline": method}))
        events = events[events["kind"].isin(("apnea", "hypopnea"))]
        out[f"rules, {method.replace('_', ' ')}"] = (events["start_time"].to_numpy(float), events["end_time"].to_numpy(float))
    output_dir = settings.run.output_dir
    if (output_dir / stage / "hmm_events.csv").exists():
        events = pd.read_csv(output_dir / stage / "hmm_events.csv")
        events = events[events["kind"].isin(("apnea", "hypopnea"))]
        out["HMM, reduced-flow state"] = (events["start_time"].to_numpy(float), events["end_time"].to_numpy(float))
    metrics_path = output_dir / "training" / "metrics.json"
    trained = {m.algorithm: m.event_clusters for m in read_metrics(metrics_path)} if metrics_path.exists() else {}
    for model in models:
        path = output_dir / stage / "csvs" / f"{model}.csv"
        if trained.get(model) and path.exists():
            frame = pd.read_csv(path)
            out[f"{model}, event clusters"] = window_intervals(frame["window_start_time"], frame["window_end_time"], frame["cluster_id"].isin(trained[model]))
    paths = [output_dir / s / "csvs" / f"{models[0]}.csv" for s in STAGES] if models else []
    if stage == "validation" and paths and all(p.exists() for p in paths):
        train, validation = (pd.read_csv(p) for p in paths)
        _, probability = event_classifier(train, validation)
        out["XGBoost on rule labels"] = window_intervals(validation["window_start_time"], validation["window_end_time"], probability >= 0.5)
    return out


def oximetry_comparison(settings: Settings, recordings: dict[str, Recording], models: list[str]) -> pd.DataFrame:
    """Score every detector against the SpO2 desaturations of each night; writes comparison/oximetry.{csv,png} and oximetry_scores.png."""
    rows, comparison = [], settings.run.output_dir / "comparison"
    for stage, rec in recordings.items():
        path = settings.run.output_dir / stage / "desaturations.csv"
        if not path.exists():
            continue
        desaturations = pd.read_csv(path)
        segment = settings.segment(stage)
        hypnogram = load_hypnogram(segment.hypnogram_path) if segment.hypnogram_path else None
        hours = (hypnogram["stage"] != "W").sum() * 30 / 3600 if hypnogram is not None else rec.signal.duration / 3600
        span = (rec.signal.start_time, rec.signal.start_time + rec.signal.duration)
        for method, (starts, ends) in detector_intervals(settings, rec, models, stage).items():
            scores = desaturation_scores(starts, ends, desaturations, settings.oximetry, span, hypnogram)
            rows.append({"stage": stage, "method": method, "events_per_hour": scores.pop("n_events") / max(hours, 1e-9), **scores})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table.to_csv(comparison / "oximetry.csv", index=False)
    save_oximetry_scores(table, comparison / "oximetry_scores.png")
    oximetry_figure(settings, recordings)
    return table


def oximetry_figure(settings: Settings, recordings: dict[str, Recording], seconds: float = 600.0) -> Path | None:
    """Ten minutes with the most confirmed events in the last recording: airflow events (confirmed or not) above SpO2 with the desaturation nadirs."""
    stage, rec = list(recordings.items())[-1]
    spo2, path = load_spo2(settings, stage), settings.run.output_dir / stage / "desaturations.csv"
    if spo2 is None or not path.exists():
        return None
    envelope = flow_envelope(rec.signal, settings.data.smoothing_seconds)
    events = rec.events
    linked = events[events["desaturation"].notna()] if "desaturation" in events.columns else events
    starts = np.arange(rec.signal.start_time, rec.signal.start_time + rec.signal.duration - seconds + 1, 300.0)
    busiest = max(starts, key=lambda t: ((linked["start_time"] >= t) & (linked["start_time"] < t + seconds)).sum()) if len(starts) else rec.signal.start_time
    out = settings.run.output_dir / "comparison" / "oximetry.png"
    save_oximetry_excerpt(envelope, events, spo2, pd.read_csv(path), float(np.median(envelope["envelope"])), float(busiest), seconds,
                          settings.oximetry.lag_seconds, f"Airflow events and SpO2 ({stage}, 10 minutes with most confirmed events)", out)
    return out


# ------------------------------------------------------------------ hidden Markov model
def hmm_agreement(output_dir: Path, model: str) -> pd.DataFrame:
    """Window precision / recall / F1 of the HMM events against the rule-based event windows (comparable to the clusterers' event F1)."""
    rows = []
    for stage in STAGES:
        paths = output_dir / stage / "csvs" / f"{model}.csv", output_dir / stage / "hmm_events.csv"
        if not all(p.exists() for p in paths):
            continue
        frame, events = pd.read_csv(paths[0]), pd.read_csv(paths[1])
        starts, ends = frame["window_start_time"].to_numpy(float), frame["window_end_time"].to_numpy(float)
        overlap = (frame["window_end_time"] - frame["window_start_time"]).median() / 6  # same 5 s of a 30 s window as the rule labels
        predicted = np.isin(window_event_labels(starts, ends, events, overlap), ["apnea", "hypopnea"])
        actual = frame["event"].isin(["apnea", "hypopnea"]).to_numpy()
        precision, recall = (predicted & actual).sum() / max(predicted.sum(), 1), (predicted & actual).sum() / max(actual.sum(), 1)
        rows.append({"stage": stage, "event_f1": 2 * precision * recall / max(precision + recall, 1e-12), "event_precision": precision, "event_recall": recall})
    return pd.DataFrame(rows)
