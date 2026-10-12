"""Static (PNG) and interactive (HTML) plots written to disk; nothing pops up."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from plotly.colors import qualitative  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from psng.data import SignalData  # noqa: E402

plt.rcParams.update({  # one clean house style for every figure
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "font.family": "DejaVu Sans", "figure.titlesize": 12, "figure.titleweight": "semibold",
})

EVENT_COLORS = {"apnea": "#c0392b", "hypopnea": "#e67e22", "artefact": "#95a5a6"}
STAGE_LEVEL = {"N3": 0, "N2": 1, "N1": 2, "REM": 3, "W": 4}
STAGE_COLORS = {"training": "#2c7fb8", "validation": "#41ab5d"}


def cluster_color(k: int) -> tuple:
    return plt.get_cmap("tab20")(k % 20)


def cluster_name(k: int, names: dict[int, str] | None) -> str:
    if names and int(k) in names:
        return f"{k}: {names[int(k)]}" if k >= 0 else names[int(k)]
    return "noise" if k < 0 else f"cluster {k}"


# ------------------------------------------------------------------ per model
def save_scatter(X: np.ndarray, labels: np.ndarray, centers: np.ndarray | None, title: str, path: Path, names: dict[int, str] | None = None) -> None:
    """2-D PCA projection of windows coloured by cluster."""
    if X.shape[1] < 2:
        X = np.c_[X, np.zeros(len(X))]
        centers = None if centers is None else np.c_[centers, np.zeros(len(centers))]
    pca = PCA(n_components=2).fit(X)
    points = pca.transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in np.unique(labels):
        m = labels == k
        ax.scatter(points[m, 0], points[m, 1], s=12, alpha=0.6, color="black" if k < 0 else cluster_color(k), label=cluster_name(k, names))
    if centers is not None and len(centers):
        c = pca.transform(centers)
        ax.scatter(c[:, 0], c[:, 1], c="black", marker="x", s=150, linewidths=3, label="centers")
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_timeline(frame: pd.DataFrame, events: pd.DataFrame, title: str, path: Path, names: dict[int, str] | None = None) -> None:
    """Cluster of every window over the night, with detected events shaded, plus the hypnogram when present."""
    has_stages = "stage" in frame.columns and frame["stage"].ne("?").any()
    fig, axes = plt.subplots(2 if has_stages else 1, 1, figsize=(16, 6 if has_stages else 4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]} if has_stages else None, squeeze=False)
    ax = axes[0, 0]
    hours = frame["window_start_time"] / 3600
    width = (frame["window_end_time"] - frame["window_start_time"]) / 3600
    for k in sorted(frame["cluster_id"].unique()):
        m = frame["cluster_id"] == k
        ax.bar(hours[m], 0.8, width=width[m], bottom=k - 0.4, align="edge", color="black" if k < 0 else cluster_color(k), linewidth=0)
    _shade_events(ax, events)
    ax.set(title=title, ylabel="cluster")
    ids = sorted(frame["cluster_id"].unique())
    ax.set_yticks(ids, [cluster_name(k, names) for k in ids], fontsize=8)
    ax.legend(handles=[Patch(color=c, alpha=0.35, label=k) for k, c in EVENT_COLORS.items()], loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    if has_stages:
        _plot_window_stages(axes[1, 0], frame)
    axes[-1, 0].set_xlabel("Time (h)")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_interactive(signal: SignalData, frame: pd.DataFrame, title: str, path: Path, max_points: int = 100_000,
                     events: pd.DataFrame | None = None, hypnogram: pd.DataFrame | None = None, max_shapes: int = 600,
                     names: dict[int, str] | None = None) -> None:
    """Signal coloured by cluster (one toggleable trace per cluster), windows-per-cluster bars, event shading and hypnogram.

    The signal is decimated to at most `max_points` samples to keep the HTML small.
    """
    step = max(1, len(signal.signal) // max_points)
    y = signal.signal[::step]
    t = signal.start_time + np.arange(len(signal.signal))[::step] / signal.sfreq
    cluster_of_sample = np.full(len(y), -2)
    starts = ((frame["window_start_time"] - signal.start_time) * signal.sfreq / step).round().astype(int)
    ends = ((frame["window_end_time"] - signal.start_time) * signal.sfreq / step).round().astype(int)
    for s, e, k in zip(starts, ends, frame["cluster_id"]):
        cluster_of_sample[s:e] = k

    ids = sorted(frame["cluster_id"].unique())
    colors = [qualitative.Dark24[i % len(qualitative.Dark24)] for i in range(len(ids))]
    with_hypnogram = hypnogram is not None and len(hypnogram) > 0
    rows = 3 if with_hypnogram else 2
    fig = make_subplots(rows=rows, cols=1, row_heights=[0.6, 0.2, 0.2][:rows], vertical_spacing=0.08,
                        subplot_titles=("Clustered signal (events shaded: red apnea, orange hypopnea, grey artefact)", "Windows per cluster", "Hypnogram")[:rows])
    fig.add_trace(go.Scattergl(x=t, y=y, name="signal", line=dict(color="lightgray", width=1), hoverinfo="skip"), row=1, col=1)
    for color, k in zip(colors, ids):
        idx = np.flatnonzero(cluster_of_sample == k)
        breaks = np.flatnonzero(np.diff(idx) > 1) + 1  # NaN separators between non-adjacent windows
        fig.add_trace(go.Scattergl(x=np.insert(t[idx], breaks, np.nan), y=np.insert(y[idx], breaks, np.nan),
                                   name=cluster_name(k, names), line=dict(color=color, width=1.5)), row=1, col=1)
    counts = frame["cluster_id"].value_counts().sort_index()
    fig.add_trace(go.Bar(x=[cluster_name(k, names) for k in counts.index], y=counts.values, marker_color=colors,
                         text=counts.values, textposition="auto", showlegend=False), row=2, col=1)
    always = [0, len(ids) + 1]
    if with_hypnogram:
        level = hypnogram["stage"].map(STAGE_LEVEL)
        fig.add_trace(go.Scatter(x=hypnogram["end_time"], y=level, mode="lines", line=dict(shape="hv", color="#444"), name="stage",
                                 text=hypnogram["stage"], hovertemplate="%{text} at %{x:.0f}s"), row=3, col=1)
        always.append(len(ids) + 2)
        fig.update_yaxes(tickvals=list(STAGE_LEVEL.values()), ticktext=list(STAGE_LEVEL), row=3, col=1)
    if events is not None and len(events):
        shown = events.sort_values("duration", ascending=False).head(max_shapes)
        for row in shown.itertuples():
            fig.add_vrect(x0=row.start_time, x1=row.end_time, fillcolor=EVENT_COLORS[row.kind], opacity=0.25, line_width=0, row=1, col=1)

    n_traces = len(fig.data)
    buttons = [dict(label="All", method="restyle", args=[{"visible": [True] * n_traces}])]
    for i, k in enumerate(ids):
        visible = [j in always or j == i + 1 for j in range(n_traces)]
        buttons.append(dict(label=cluster_name(k, names), method="restyle", args=[{"visible": visible}]))
    fig.update_layout(title=title, height=950, dragmode="pan",
                      updatemenus=[dict(type="buttons", direction="right", buttons=buttons, x=0, y=1.08, xanchor="left")])
    fig.update_xaxes(title_text="Time (s)", rangeslider=dict(visible=True, thickness=0.05), row=1, col=1)
    fig.update_yaxes(title_text=signal.channel, row=1, col=1)
    fig.write_html(path, include_plotlyjs=True)


# ------------------------------------------------------------------ per recording
BASELINE_STYLE = {"rolling_median": ("#1a1a1a", "-"), "kalman": ("#8e44ad", "--")}


def save_events_overview(envelope: pd.DataFrame, events: pd.DataFrame, hypnogram: pd.DataFrame | None, baseline: float, title: str, path: Path,
                         baselines: dict[str, np.ndarray] | None = None) -> None:
    """Airflow envelope relative to the recording baseline with detected events shaded, the local baseline trackers and the hypnogram."""
    with_hypnogram = hypnogram is not None and len(hypnogram) > 0
    rows = 3 if with_hypnogram else 2
    fig, axes = plt.subplots(rows, 1, figsize=(16, 7 if with_hypnogram else 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 0.8, 1][:rows]}, squeeze=False)
    hours = envelope["time"] / 3600
    ratio = envelope["envelope"] / max(baseline, 1e-12)
    smooth = ratio.rolling(31, center=True, min_periods=1).median()  # 30 s median: readable over a whole night
    ax = axes[0, 0]
    ax.fill_between(hours, 0, smooth.clip(upper=2.5), color="#9ecae1", alpha=0.6, linewidth=0)
    ax.plot(hours, smooth.clip(upper=2.5), color="#3182bd", linewidth=0.7)
    handles = [Patch(color="#9ecae1", label="airflow envelope / night median (30 s median)")]
    for name, values in (baselines or {}).items():
        color, style = BASELINE_STYLE.get(name, ("black", ":"))
        ax.plot(hours, values / max(baseline, 1e-12), color=color, linestyle=style, linewidth=1.1)
        handles.append(plt.Line2D([], [], color=color, linestyle=style, label=f"{name.replace('_', ' ')} baseline"))
    ax.set(title=title, ylabel="relative amplitude", ylim=(0, 2.5))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    strip = axes[1, 0]  # one tick per event, so short events stay visible
    counts = events["kind"].value_counts() if len(events) else {}
    for i, (kind, color) in enumerate(EVENT_COLORS.items()):
        rows_k = events[events["kind"] == kind]
        strip.vlines((rows_k["start_time"] + rows_k["end_time"]) / 7200, i - 0.4, i + 0.4, color=color, linewidth=1.2)
    strip.set_yticks(range(len(EVENT_COLORS)), [f"{k} ({counts.get(k, 0)})" for k in EVENT_COLORS])
    strip.set_ylim(-0.6, len(EVENT_COLORS) - 0.4)
    strip.grid(False)
    if with_hypnogram:
        ax2 = axes[2, 0]
        ax2.step(hypnogram["end_time"] / 3600, hypnogram["stage"].map(STAGE_LEVEL), where="pre", color="#444", linewidth=1)
        ax2.set_yticks(list(STAGE_LEVEL.values()), list(STAGE_LEVEL))
        ax2.set_ylabel("stage")
    axes[-1, 0].set_xlim(hours.min(), hours.max())
    axes[-1, 0].set_xlabel("Time (h)")
    fig.tight_layout(rect=(0, 0, 0.84, 1))  # room for the legend on the right
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _shade_events(ax, events: pd.DataFrame) -> None:
    for row in events.itertuples():
        ax.axvspan(row.start_time / 3600, row.end_time / 3600, color=EVENT_COLORS.get(row.kind, "grey"), alpha=0.35, linewidth=0)


def _plot_window_stages(ax, frame: pd.DataFrame) -> None:
    level = frame["stage"].map(STAGE_LEVEL)
    ax.step(frame["window_start_time"] / 3600, level, where="post", color="#444", linewidth=1)
    ax.set_yticks(list(STAGE_LEVEL.values()), list(STAGE_LEVEL))
    ax.set_ylabel("stage")


# ------------------------------------------------------------------ comparisons across models and stages
COMPARISON_METRICS = [
    ("silhouette", "Silhouette (higher = better)"),
    ("davies_bouldin", "Davies-Bouldin (lower = better)"),
    ("calinski_harabasz", "Calinski-Harabasz (higher = better)"),
    ("event_f1", "Event F1 vs rule-based events"),
    ("event_nmi", "NMI vs event labels"),
    ("stage_nmi", "NMI vs sleep stages"),
]


def save_comparison(metrics: pd.DataFrame, path: Path) -> None:
    """Grouped bars per algorithm, training next to validation, one panel per metric."""
    algorithms = list(dict.fromkeys(metrics["algorithm"]))
    stages = [s for s in ("training", "validation") if s in set(metrics["stage"])]
    panels = [(m, label) for m, label in COMPARISON_METRICS if m in metrics.columns and metrics[m].notna().any()]
    ncols = 3
    nrows = max(1, int(np.ceil(len(panels) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(len(algorithms))
    width = 0.8 / max(1, len(stages))
    for ax, (metric, label) in zip(axes.flat, panels):
        for i, stage in enumerate(stages):
            rows = metrics[metrics["stage"] == stage].set_index("algorithm").reindex(algorithms)
            values = rows[metric].to_numpy(float)
            bars = ax.bar(x + (i - (len(stages) - 1) / 2) * width, np.nan_to_num(values), width, label=stage, color=STAGE_COLORS.get(stage))
            for bar, v in zip(bars, values):
                if np.isfinite(v):
                    ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(x, algorithms, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Clustering quality per algorithm: training vs validation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_event_rates(profiles: pd.DataFrame, path: Path) -> None:
    """Share of apnea / hypopnea windows in each cluster, per algorithm, training next to validation."""
    algorithms = list(dict.fromkeys(profiles["algorithm"]))
    stages = [s for s in ("training", "validation") if s in set(profiles["stage"])]
    ncols = min(4, len(algorithms))
    nrows = int(np.ceil(len(algorithms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.3 * nrows), squeeze=False)
    for ax, algorithm in zip(axes.flat, algorithms):
        rows = profiles[profiles["algorithm"] == algorithm]
        ids = sorted(rows["cluster_id"].unique())
        x = np.arange(len(ids))
        width = 0.8 / max(1, len(stages))
        for i, stage in enumerate(stages):
            r = rows[rows["stage"] == stage].set_index("cluster_id").reindex(ids)
            bars = ax.bar(x + (i - (len(stages) - 1) / 2) * width, r["event_rate"].fillna(0), width, label=stage, color=STAGE_COLORS.get(stage))
            for bar, n in zip(bars, r["n_windows"]):
                if np.isfinite(n):
                    ax.annotate(f"{int(n)}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=6)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
        ax.set(title=algorithm, ylim=(0, 1.1), ylabel="event window share")
        labels = rows.drop_duplicates("cluster_id").set_index("cluster_id")["name"].reindex(ids) if "name" in rows.columns else pd.Series(index=ids, dtype=object)
        ax.set_xticks(x, [str(k) if pd.isna(labels.get(k)) else f"{k}: {labels[k]}" for k in ids], fontsize=7, rotation=30, ha="right")
    for ax in axes.flat[len(algorithms):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Apnea / hypopnea share per cluster (bar labels: windows in the cluster; clusters above 0.5 count as event detectors)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_sweep(results: pd.DataFrame, model: str, path: Path) -> None:
    """Internal and event metrics as a function of the number of clusters."""
    panels = [(m, label) for m, label in COMPARISON_METRICS if m in results.columns and results[m].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), squeeze=False)
    for ax, (metric, label) in zip(axes.flat, panels):
        ax.plot(results["k"], results[metric], marker="o")
        ax.set(title=label, xlabel="clusters")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model}: metrics vs number of clusters (training)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ example time slices
def save_examples_grid(examples: dict[str, dict[str, list[dict]]], title: str, path: Path, ylim: float = 3.5) -> None:
    """Grid of signal snippets: one row per group (event kind or cluster), training examples left, validation right.

    `examples[row][stage]` is a list of slices `{"t": seconds, "y": signal in baseline units, "spans": [(start, end, kind)],
    "label": str}`; `kind` is an event kind or `window` (shaded grey).
    """
    stages = [s for s in ("training", "validation") if any(s in v for v in examples.values())]
    per_stage = max((len(v.get(s, [])) for v in examples.values() for s in stages), default=1)
    rows = list(examples)
    ncols = max(1, per_stage * len(stages))
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.1 * ncols, 1.5 * len(rows) + 0.8), squeeze=False, sharey=True)
    for r, row in enumerate(rows):
        for si, stage in enumerate(stages):
            slices = examples[row].get(stage, [])
            for j in range(per_stage):
                ax = axes[r, si * per_stage + j]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(STAGE_COLORS.get(stage, "black"))
                if j >= len(slices):
                    ax.text(0.5, 0.5, "no example", ha="center", va="center", fontsize=7, color="grey", transform=ax.transAxes)
                    continue
                sl = slices[j]
                t0 = sl["t"][0]
                ax.plot(sl["t"] - t0, np.clip(sl["y"], -ylim, ylim), color="#333", linewidth=0.6)
                for start, end, kind in sl["spans"]:
                    ax.axvspan(start - t0, end - t0, color=EVENT_COLORS.get(kind, "#bbbbbb"), alpha=0.25 if kind == "window" else 0.35, linewidth=0)
                ax.set_xlim(0, sl["t"][-1] - t0)
                ax.set_ylim(-ylim, ylim)
                ax.set_title(sl["label"], fontsize=7, pad=2)
                if si * per_stage + j == 0:
                    ax.set_ylabel(row, fontsize=8, rotation=0, ha="right", va="center")
    for si, stage in enumerate(stages):
        fig.text((si + 0.5) / len(stages), 0.995, stage, ha="center", va="top", fontsize=11, color=STAGE_COLORS.get(stage), weight="bold")
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ feature spaces side by side
def save_spaces_heatmap(results: pd.DataFrame, path: Path, metrics=(("silhouette", "Silhouette"), ("event_f1", "Event F1"), ("event_nmi", "Event NMI")),
                        reference: dict[str, float] | None = None) -> None:
    """Algorithms x feature spaces, one heat-map per metric (columns) and stage (rows)."""
    algorithms = list(dict.fromkeys(results["algorithm"]))
    spaces = list(dict.fromkeys(results["space"]))
    stages = [s for s in ("training", "validation") if s in set(results["stage"])]
    fig, axes = plt.subplots(len(stages), len(metrics), figsize=(3.2 * len(metrics) + 1, 0.45 * len(algorithms) * len(stages) + 2), squeeze=False)
    for r, stage in enumerate(stages):
        for c, (metric, label) in enumerate(metrics):
            ax = axes[r, c]
            ax.grid(False)
            table = results[results["stage"] == stage].pivot_table(index="algorithm", columns="space", values=metric, aggfunc="first")
            values = table.reindex(index=algorithms, columns=spaces).to_numpy(float)
            finite = np.isfinite(values)
            lo, hi = (np.nanmin(values), np.nanmax(values)) if finite.any() else (0, 1)
            ax.imshow(np.where(finite, values, lo), cmap="Blues", aspect="auto", vmin=lo, vmax=hi if hi > lo else lo + 1)
            for i in range(len(algorithms)):
                for j in range(len(spaces)):
                    v = values[i, j]
                    dark = finite[i, j] and hi > lo and (v - lo) / (hi - lo) > 0.6
                    ax.text(j, i, "-" if not finite[i, j] else f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if dark else "#222")
            ax.set_xticks(range(len(spaces)), spaces)
            ax.set_yticks(range(len(algorithms)), algorithms if c == 0 else [])
            title = f"{label} - {stage}"
            if reference and metric in reference and stage == "validation":
                title += f"  (supervised ref {reference[metric]:.2f})"
            ax.set_title(title)
    fig.suptitle("Feature spaces side by side: PCA vs raw features vs autoencoder")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_importances(importances: dict[str, float], title: str, path: Path) -> None:
    names = list(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1))
    ax.barh(names, [importances[n] for n in names], color="#2c7fb8")
    ax.set(title=title, xlabel="importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_embedding(points: dict[str, np.ndarray], frames: dict[str, pd.DataFrame], names: dict[int, str], title: str, path: Path) -> None:
    """2-D embedding per stage (t-SNE), coloured by cluster, event windows drawn with a red edge."""
    stages = list(points)
    fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 5.2), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        ax.grid(False)
        xy, frame = points[stage], frames[stage]
        is_event = frame["event"].isin(["apnea", "hypopnea"]).to_numpy()
        for k in sorted(frame["cluster_id"].unique()):
            m = (frame["cluster_id"] == k).to_numpy()
            ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.75, color="black" if k < 0 else cluster_color(k), label=cluster_name(k, names), linewidths=0)
        ax.scatter(xy[is_event, 0], xy[is_event, 1], s=30, facecolors="none", edgecolors=EVENT_COLORS["apnea"], linewidths=0.7, label="apnea / hypopnea window")
        ax.set(title=f"{stage}: {len(frame)} windows", xticks=[], yticks=[])
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=1.6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ method figures: Kalman, XGBoost, autoencoder
def save_kalman_excerpt(envelope: pd.DataFrame, baselines: dict[str, np.ndarray], events: dict[str, pd.DataFrame], night_median: float,
                        start: float, seconds: float, title: str, path: Path) -> None:
    """One excerpt: envelope with both baseline trackers, and the events each tracker finds (one strip per tracker)."""
    m = (envelope["time"] >= start) & (envelope["time"] < start + seconds)
    minutes = (envelope.loc[m, "time"] - start) / 60
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.fill_between(minutes, 0, envelope.loc[m, "envelope"] / night_median, color="#9ecae1", alpha=0.7, linewidth=0, label="airflow envelope (1 s RMS)")
    for name, values in baselines.items():
        color, style = BASELINE_STYLE.get(name, ("black", ":"))
        ax.plot(minutes, values[m.to_numpy()] / night_median, color=color, linestyle=style, linewidth=1.4, label=f"{name.replace('_', ' ')} baseline")
    ax.set(title=title, ylabel="amplitude / night median", ylim=(0, None))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    strip = axes[1]
    strip.grid(False)
    for i, (name, rows) in enumerate(events.items()):
        rows = rows[(rows["end_time"] > start) & (rows["start_time"] < start + seconds)]
        for r in rows.itertuples():
            strip.barh(i, (min(r.end_time, start + seconds) - max(r.start_time, start)) / 60, left=(max(r.start_time, start) - start) / 60, height=0.6,
                       color=EVENT_COLORS.get(r.kind, "grey"), linewidth=0)
    strip.set_yticks(range(len(events)), [f"{n.replace('_', ' ')} ({len(r[(r['end_time'] > start) & (r['start_time'] < start + seconds)])})" for n, r in events.items()])
    strip.set_ylim(-0.6, len(events) - 0.4)
    strip.set_xlim(0, seconds / 60)
    strip.set_xlabel(f"minutes from {int(start // 3600)}:{int(start % 3600 // 60):02d}:00")
    strip.legend(handles=[Patch(color=c, label=k) for k, c in EVENT_COLORS.items()], loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_supervised(frame: pd.DataFrame, probability: np.ndarray, auc: float | None, f1: float, title: str, path: Path) -> None:
    """Predicted event probability over the validation night (rule-based event windows marked) and the ROC curve."""
    from sklearn.metrics import roc_curve

    is_event = frame["event"].isin(["apnea", "hypopnea"]).to_numpy()
    hours = frame["window_start_time"] / 3600
    fig, (ax, roc) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [4, 1.2]})
    ax.fill_between(hours, 0, probability, step="post", color="#9ecae1", alpha=0.8, linewidth=0, label="P(event window)")
    ax.vlines(hours[is_event], 1.02, 1.1, color=EVENT_COLORS["hypopnea"], linewidth=0.8, label="rule-based event window")
    ax.axhline(0.5, color="#444", linestyle=":", linewidth=0.8, label="decision threshold")
    ax.set(title=title, xlabel="Time (h)", ylabel="probability", ylim=(0, 1.12), xlim=(hours.min(), hours.max()))
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.18))
    if auc is not None and is_event.any() and not is_event.all():
        fpr, tpr, _ = roc_curve(is_event, probability)
        roc.plot(fpr, tpr, color="#2c7fb8", linewidth=1.6, label=f"AUC {auc:.3f}")
    roc.plot([0, 1], [0, 1], color="#999", linestyle=":", linewidth=0.8)
    roc.set(title=f"ROC (F1 at 0.5 = {f1:.2f})", xlabel="false positive rate", ylabel="true positive rate", xlim=(0, 1), ylim=(0, 1))
    roc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_latent_spaces(spaces: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray]]], title: str, path: Path) -> None:
    """Rows = stages, columns = reduced spaces (e.g. PCA, autoencoder); first two components coloured by event kind."""
    kinds = {"normal": "#bdbdbd", "hypopnea": EVENT_COLORS["hypopnea"], "apnea": EVENT_COLORS["apnea"], "artefact": EVENT_COLORS["artefact"]}
    stages = list(next(iter(spaces.values())))
    fig, axes = plt.subplots(len(stages), len(spaces), figsize=(5 * len(spaces), 4.2 * len(stages)), squeeze=False)
    for c, (space, per_stage) in enumerate(spaces.items()):
        for r, stage in enumerate(stages):
            ax = axes[r, c]
            ax.grid(False)
            Z, events = per_stage[stage]
            for kind, color in kinds.items():
                m = events == kind
                if m.any():
                    ax.scatter(Z.iloc[m.nonzero()[0], 0], Z.iloc[m.nonzero()[0], 1], s=10, color=color, alpha=0.8 if kind != "normal" else 0.5, linewidths=0, label=f"{kind} ({m.sum()})")
            ax.set(title=f"{space} - {stage}", xlabel=Z.columns[0], ylabel=Z.columns[1])
            ax.legend(markerscale=1.8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
