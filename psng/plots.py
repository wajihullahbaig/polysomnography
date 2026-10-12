"""Static (PNG) and interactive (HTML) plots written to disk; nothing pops up."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
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
NIGHT = {"training": "night 1 (training)", "validation": "night 2 (validation)"}
# sleep depth is ordinal: one blue ramp from awake (light) to deep sleep (dark); REM is its own hue
STAGE_BAND = {"W": "#ececec", "N1": "#c6dbef", "N2": "#6baed6", "N3": "#08519c", "REM": "#b4a7d6"}
STAGE_NAMES = {"W": "awake", "N1": "drowsy (N1)", "N2": "light sleep (N2)", "N3": "deep sleep (N3)", "REM": "dream sleep (REM)"}
FLOW_COLOR, FLOW_FILL = "#3182bd", "#9ecae1"
# cluster colours follow what the cluster means, so orange / red always reads as "breathing events" across figures
_MEANING = [("artefact", "#95a5a6"), ("hypopnea", EVENT_COLORS["hypopnea"]), ("apnea", EVENT_COLORS["apnea"]), ("with events", "#f0b27a"),
            ("reduced airflow", "#f5cba7"), ("irregular", "#8e7cc3"), ("large breaths", "#2c7fb8"), ("slow", "#17a2b8"), ("fast", "#17a2b8"),
            ("regular breathing", "#7f9fb5")]
_FALLBACK = ["#7f9fb5", "#8e7cc3", "#2c7fb8", "#17a2b8", "#8c6d31", "#636363"]


def cluster_color(k: int, names: dict[int, str] | None = None) -> str:
    """Colour by meaning of the cluster name (events warm, breathing cool); repeated names get lighter shades."""
    if k < 0:
        return "#cfcfcf"
    name = (names or {}).get(int(k), "")
    base = next((c for key, c in _MEANING if key in name), _FALLBACK[int(k) % len(_FALLBACK)])
    repeat = int(re.search(r"#(\d+)$", name).group(1)) - 1 if re.search(r"#(\d+)$", name) else 0
    mix = min(0.25 * repeat, 0.6)
    return to_hex(tuple(c + (1 - c) * mix for c in to_rgb(base)))


def _header(fig, title: str, subtitle: str | None = None, pad: float = 0.0) -> None:
    """Bold title and a grey 'what to look at' line, left-aligned above the figure."""
    height = fig.get_size_inches()[1]
    top_axes = max(fig.axes, key=lambda ax: ax.get_position().y1, default=None)  # call after the layout is final
    top = (top_axes.get_position().y1 if top_axes is not None else 1.0) + pad / height  # pad: inches of text drawn above the top axes
    if top_axes is not None and top_axes.get_title():
        top += 0.3 / height
    fig.text(0.01, top + 0.5 / height, title, ha="left", va="bottom", fontsize=13, weight="semibold", color="#1a1a1a")
    if subtitle:
        fig.text(0.01, top + 0.15 / height, subtitle, ha="left", va="bottom", fontsize=9.5, color="#555555")


def _band(ax, starts, ends, labels, colors: dict, y: float = 0.0, height: float = 1.0) -> None:
    """One horizontal band of coloured segments (time in hours)."""
    starts, ends, labels = np.asarray(starts, float), np.asarray(ends, float), np.asarray(labels, dtype=object)
    for label in pd.unique(labels):
        m = labels == label
        if label in colors:
            ax.broken_barh(list(zip(starts[m], ends[m] - starts[m])), (y, height), facecolors=colors[label], linewidth=0)


def _band_axis(ax, label: str) -> None:
    ax.set_yticks([0.5], [label])
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)


def _stage_band(ax, starts, ends, stages, legend: bool = True) -> None:
    _band(ax, starts, ends, stages, STAGE_BAND)
    _band_axis(ax, "sleep stage")
    if legend:
        present = [k for k in ("W", "REM", "N1", "N2", "N3") if k in set(stages)]
        ax.legend(handles=[Patch(color=STAGE_BAND[k], label=STAGE_NAMES[k]) for k in present], loc="upper left", bbox_to_anchor=(1.01, 1.4), borderaxespad=0)


def _hypnogram_band(ax, hypnogram: pd.DataFrame, legend: bool = True) -> None:
    epoch = float(np.median(np.diff(hypnogram["end_time"]))) if len(hypnogram) > 1 else 30.0
    _stage_band(ax, (hypnogram["end_time"] - epoch) / 3600, hypnogram["end_time"] / 3600, hypnogram["stage"].to_numpy(object), legend)


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
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.grid(False)
    for k in np.unique(labels):
        m = labels == k
        ax.scatter(points[m, 0], points[m, 1], s=14, alpha=0.75, color=cluster_color(k, names), linewidths=0, label=cluster_name(k, names))
    if centers is not None and len(centers):
        c = pca.transform(centers)
        ax.scatter(c[:, 0], c[:, 1], c="#1a1a1a", marker="x", s=110, linewidths=2.2, label="group centres")
    ax.set(xlabel="main direction 1 (PC1)", ylabel="main direction 2 (PC2)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, markerscale=1.6)
    _header(fig, title, "Each dot is a 30 s window; nearby dots breathe alike. Warm colours = groups dominated by breathing events.")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_timeline(frame: pd.DataFrame, events: pd.DataFrame, title: str, path: Path, names: dict[int, str] | None = None) -> None:
    """Three bands over the night: cluster of every window, rule-based events, and the sleep stage when present."""
    has_stages = "stage" in frame.columns and frame["stage"].ne("?").any()
    rows = 3 if has_stages else 2
    fig, axes = plt.subplots(rows, 1, figsize=(15, 0.9 * rows + 1.2), sharex=True, squeeze=False)
    axes = axes[:, 0]
    starts, ends = frame["window_start_time"] / 3600, frame["window_end_time"] / 3600
    ids = sorted(frame["cluster_id"].unique())
    colors = {k: cluster_color(k, names) for k in ids}
    _band(axes[0], starts, ends, frame["cluster_id"].to_numpy(), colors)
    _band_axis(axes[0], "group")
    shares = frame["cluster_id"].value_counts(normalize=True)
    axes[0].legend(handles=[Patch(color=colors[k], label=f"{cluster_name(k, names)} ({shares.get(k, 0):.0%})") for k in ids],
                   loc="upper left", bbox_to_anchor=(1.01, 1.2), borderaxespad=0)
    kinds = events[events["kind"].isin(EVENT_COLORS)] if len(events) else events
    if len(kinds):
        _band(axes[1], kinds["start_time"] / 3600, kinds["end_time"] / 3600, kinds["kind"].to_numpy(object), EVENT_COLORS)
    _band_axis(axes[1], "events (rules)")
    counts = kinds["kind"].value_counts() if len(kinds) else {}
    axes[1].legend(handles=[Patch(color=c, label=f"{k} ({counts.get(k, 0)})") for k, c in EVENT_COLORS.items()],
                   loc="upper left", bbox_to_anchor=(1.01, 1.2), borderaxespad=0)
    if has_stages:
        known = frame["stage"].ne("?").to_numpy()
        _stage_band(axes[2], starts[known], ends[known], frame["stage"].to_numpy(object)[known])
    axes[-1].set_xlabel("hours since the recording started")
    axes[-1].set_xlim(starts.min(), ends.max())
    fig.subplots_adjust(hspace=0.35)
    _header(fig, title, "Top: the group each 30 s window falls in. Middle: breathing events found by the rules. Bottom: sleep stage.")
    fig.savefig(path, dpi=140, bbox_inches="tight")
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
    colors = [cluster_color(k, names) for k in ids]
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
                         baselines: dict[str, np.ndarray] | None = None, spo2: pd.DataFrame | None = None, bin_minutes: float = 10.0) -> None:
    """A night at a glance: breathing strength, events per 10 minutes, blood oxygen and sleep stage on one time axis."""
    panels = ["flow", "events"] + (["spo2"] if spo2 is not None and spo2["spo2"].notna().any() else []) + (["stage"] if hypnogram is not None and len(hypnogram) else [])
    heights = {"flow": 2.2, "events": 1.4, "spo2": 1.5, "stage": 0.45}
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, sum(heights[k] for k in panels) + 1.2), sharex=True,
                             gridspec_kw={"height_ratios": [heights[k] for k in panels]}, squeeze=False)
    ax = dict(zip(panels, axes[:, 0]))
    hours = envelope["time"] / 3600
    ratio = (envelope["envelope"] / max(baseline, 1e-12)).rolling(61, center=True, min_periods=1).median()  # 1 min median: readable over a night
    ax["flow"].fill_between(hours, 0, ratio.clip(upper=2.5), color=FLOW_FILL, alpha=0.7, linewidth=0)
    ax["flow"].plot(hours, ratio.clip(upper=2.5), color=FLOW_COLOR, linewidth=0.7)
    handles = [Patch(color=FLOW_FILL, label="breathing strength (1 min median)")]
    for name, values in (baselines or {}).items():
        color, style = BASELINE_STYLE.get(name, ("black", ":"))
        ax["flow"].plot(hours, values / max(baseline, 1e-12), color=color, linestyle=style, linewidth=1.1)
        handles.append(plt.Line2D([], [], color=color, linestyle=style, label=f"normal level ({name.replace('_', ' ')})"))
    ax["flow"].axhline(1, color="#999", linewidth=0.6, linestyle=":")
    ax["flow"].set(ylabel="breathing vs typical\n(1 = typical night)", ylim=(0, 2.5))
    ax["flow"].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    edges = np.arange(hours.min(), hours.max() + bin_minutes / 60, bin_minutes / 60)
    bottom = np.zeros(len(edges) - 1)
    for kind in ("apnea", "hypopnea", "artefact"):
        rows = events[events["kind"] == kind] if len(events) else events
        counts = np.histogram(rows["start_time"] / 3600, edges)[0] if len(rows) else np.zeros(len(edges) - 1)
        ax["events"].bar(edges[:-1], counts, width=np.diff(edges) * 0.92, bottom=bottom, align="edge", color=EVENT_COLORS[kind], linewidth=0,
                         label=f"{kind} ({len(rows)})")
        bottom += counts
    ax["events"].set(ylabel=f"events per\n{bin_minutes:g} min")
    ax["events"].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    if "spo2" in ax:
        level = spo2.set_index(spo2["time"] / 3600)["spo2"].rolling(60, center=True, min_periods=10).min()
        ax["spo2"].fill_between(level.index, level.clip(lower=80), 100, color="#fde0c5", linewidth=0)
        ax["spo2"].plot(level.index, level.clip(lower=80), color="#1a1a1a", linewidth=0.7)
        ax["spo2"].axhline(90, color=EVENT_COLORS["apnea"], linewidth=0.7, linestyle=":")
        ax["spo2"].set(ylabel="blood oxygen\nSpO2 (%)", ylim=(80, 100))
        ax["spo2"].legend(handles=[plt.Line2D([], [], color="#1a1a1a", label="lowest SpO2 per minute"),
                                   plt.Line2D([], [], color=EVENT_COLORS["apnea"], linestyle=":", label="90 %")],
                          loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    if "stage" in ax:
        _hypnogram_band(ax["stage"], hypnogram)
    axes[-1, 0].set_xlim(hours.min(), hours.max())
    axes[-1, 0].set_xlabel("hours since the recording started")
    fig.subplots_adjust(hspace=0.25)
    _header(fig, title, "Breathing dips (top) show up as events (orange / red bars); oxygen falls with them. Dark blue at the bottom = deep sleep.")
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
    ("event_f1", "Finds breathing events (event F1)"),
    ("event_nmi", "Agrees with event labels (NMI)"),
    ("stage_nmi", "Agrees with sleep stages (NMI)"),
    ("silhouette", "Groups well separated (silhouette)"),
    ("davies_bouldin", "Groups overlap (Davies-Bouldin, lower = better)"),
    ("calinski_harabasz", "Groups compact (Calinski-Harabasz)"),
]


def save_comparison(metrics: pd.DataFrame, path: Path) -> None:
    """Score table as a heat-map: one row per algorithm, one column per metric and night; darker = better within the column."""
    algorithms = list(dict.fromkeys(metrics["algorithm"]))
    stages = [s for s in ("training", "validation") if s in set(metrics["stage"])]
    panels = [(m, label) for m, label in COMPARISON_METRICS if m in metrics.columns and metrics[m].notna().any()]
    columns = [(m, label, stage) for m, label in panels for stage in stages]
    values = np.array([[metrics[(metrics["algorithm"] == a) & (metrics["stage"] == st)][m].astype(float).mean() for m, _, st in columns]
                       for a in algorithms])
    shade = np.zeros_like(values)
    for j, (m, _, _) in enumerate(columns):  # rank within each column: 1 = best
        col = values[:, j]
        ok = np.isfinite(col)
        if ok.sum() > 1 and np.nanmax(col) > np.nanmin(col):
            scaled = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col))
            shade[:, j] = np.where(ok, 1 - scaled if m == "davies_bouldin" else scaled, 0)
    fig, ax = plt.subplots(figsize=(1.15 * len(columns) + 2.5, 0.5 * len(algorithms) + 2.2))
    ax.grid(False)
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        best = np.isfinite(v) and shade[i, j] == 1
        ax.text(j, i, "-" if not np.isfinite(v) else f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                color="white" if shade[i, j] > 0.6 else "#1a1a1a", weight="bold" if best else "normal")
    ax.set_xticks(range(len(columns)), ["night 1" if st == "training" else "night 2" for _, _, st in columns], fontsize=8)
    ax.set_yticks(range(len(algorithms)), algorithms)
    ax.tick_params(length=0)
    for j in range(0, len(columns), len(stages)):  # metric name above each pair of nights
        label = columns[j][1].replace(" (", "\n(")
        ax.text(j + (len(stages) - 1) / 2, -0.75, label, ha="center", va="bottom", fontsize=8.5, weight="semibold" if j == 0 else "normal")
        if j:
            ax.axvline(j - 0.5, color="white", linewidth=4)
    for side in ax.spines.values():
        side.set_visible(False)
    _header(fig, "Which grouping method works best?",
            "Darker = better within each column; best in bold. The first column (finding breathing events) is the one that matters for the goal.", pad=0.55)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_event_rates(profiles: pd.DataFrame, path: Path) -> None:
    """Per algorithm and cluster: share of windows that hold an apnea / hypopnea, night 1 (hollow) and night 2 (filled)."""
    algorithms = list(dict.fromkeys(profiles["algorithm"]))
    ncols = min(3, len(algorithms))
    nrows = int(np.ceil(len(algorithms) / ncols))
    per_panel = max(profiles.groupby("algorithm")["cluster_id"].nunique().max(), 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, (0.38 * per_panel + 1.1) * nrows), squeeze=False)
    for ax, algorithm in zip(axes.flat, algorithms):
        rows = profiles[profiles["algorithm"] == algorithm]
        named = rows.drop_duplicates("cluster_id").set_index("cluster_id")
        names = {int(k): str(v) for k, v in named["name"].items()} if "name" in named.columns else {}
        ids = sorted(rows["cluster_id"].unique(), key=lambda k: -rows.loc[rows["cluster_id"] == k, "event_rate"].mean())
        y = np.arange(len(ids))[::-1]
        ax.axvspan(0.5, 1, color="#fdebd0", linewidth=0)
        for yi, k in zip(y, ids):
            r = rows[rows["cluster_id"] == k].set_index("stage")["event_rate"]
            color = cluster_color(k, names)
            if {"training", "validation"} <= set(r.index):
                ax.plot([r["training"], r["validation"]], [yi, yi], color=color, linewidth=2, alpha=0.5)
            if "training" in r.index:
                ax.scatter(r["training"], yi, s=46, facecolors="white", edgecolors=color, linewidths=1.8, zorder=3)
            if "validation" in r.index:
                ax.scatter(r["validation"], yi, s=50, color=color, edgecolors="white", linewidths=1, zorder=4)
        ax.set_yticks(y, [cluster_name(k, names) for k in ids], fontsize=8)
        ax.set(title=algorithm, xlim=(0, 1), xlabel="share of the group's windows with an event")
        ax.grid(axis="y", visible=False)
    for ax in axes.flat[len(algorithms):]:
        ax.axis("off")
    fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#555", label="night 1"),
                        plt.Line2D([], [], marker="o", linestyle="", color="#555", label="night 2"),
                        Patch(color="#fdebd0", label="mostly events: the group acts as event detector")],
               loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)
    fig.tight_layout()
    _header(fig, "How much of each group is breathing events?",
            "Groups in the shaded area are mostly apnea / hypopnea and are used as the event detector. Similar dots on both nights = the group transfers.")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_sweep(results: pd.DataFrame, model: str, path: Path) -> None:
    """Every metric as a function of the number of clusters (training night)."""
    panels = [(m, label) for m, label in COMPARISON_METRICS if m in results.columns and results[m].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3), squeeze=False)
    for ax, (metric, label) in zip(axes.flat, panels):
        ax.plot(results["k"], results[metric], color="#2c7fb8", linewidth=2, marker="o", markersize=6, markeredgecolor="white")
        best = results.loc[results[metric].idxmin() if metric == "davies_bouldin" else results[metric].idxmax()]
        ax.scatter(best["k"], best[metric], s=90, facecolors="none", edgecolors=EVENT_COLORS["hypopnea"], linewidths=2, zorder=3)
        ax.set_title(label.replace(" (", "\n("), fontsize=9)
        ax.set(xlabel="number of groups")
    fig.tight_layout()
    _header(fig, f"How many groups should {model} use?", "Each panel scores 2-12 groups on night 1; the orange ring marks the best value.")
    fig.savefig(path, dpi=140, bbox_inches="tight")
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
        fig.text((si + 0.5) / len(stages), 0.995, NIGHT.get(stage, stage), ha="center", va="top", fontsize=11, color=STAGE_COLORS.get(stage), weight="bold")
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ feature spaces side by side
def save_spaces_heatmap(results: pd.DataFrame, path: Path, metrics=(("event_f1", "Finds events (F1)"), ("event_nmi", "Agrees with events (NMI)"),
                                                                   ("silhouette", "Separation (silhouette)")),
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
            title = f"{label} - {NIGHT.get(stage, stage)}"
            if reference and metric in reference and stage == "validation":
                title += f"\n(supervised reference {reference[metric]:.2f})"
            ax.set_title(title)
    fig.tight_layout()
    _header(fig, "Does compressing the features help?",
            "Each cell scores one grouping method on one version of the features: PCA (3 numbers), raw (all 12), autoencoder (3). Darker = higher.")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_importances(importances: dict[str, float], title: str, path: Path) -> None:
    names = list(importances)[::-1]
    values = np.array([importances[n] for n in names])
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(names) + 1.2))
    ax.barh(names, values, color=["#2c7fb8" if v >= np.sort(values)[-4] else "#bcd3e6" for v in values], height=0.7)
    for yi, v in enumerate(values):
        ax.text(v, yi, f" {v:.2f}", va="center", fontsize=8, color="#444")
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="share of the model's decisions")
    _header(fig, title, "Longer bar = the model leans on this measurement more; the top four are highlighted.")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_embedding(points: dict[str, np.ndarray], frames: dict[str, pd.DataFrame], names: dict[int, str], title: str, path: Path) -> None:
    """2-D embedding per stage (t-SNE), coloured by cluster, event windows drawn with a dark ring."""
    stages = list(points)
    fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 5.2), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        xy, frame = points[stage], frames[stage]
        is_event = frame["event"].isin(["apnea", "hypopnea"]).to_numpy()
        for k in sorted(frame["cluster_id"].unique()):
            m = (frame["cluster_id"] == k).to_numpy()
            ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.8, color=cluster_color(k, names), label=cluster_name(k, names), linewidths=0)
        ax.scatter(xy[is_event, 0], xy[is_event, 1], s=30, facecolors="none", edgecolors="#7b241c", linewidths=0.6, label="window with an event (rules)")
        ax.set(title=f"{NIGHT.get(stage, stage)}: {len(frame)} windows", xticks=[], yticks=[])
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=1.6, borderaxespad=0)
    fig.tight_layout()
    _header(fig, title, "Each dot is a 30 s window placed so that similar breathing sits close together. Ringed dots hold an apnea / hypopnea.")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ method figures: Kalman, XGBoost, autoencoder
def save_kalman_excerpt(envelope: pd.DataFrame, baselines: dict[str, np.ndarray], events: dict[str, pd.DataFrame], night_median: float,
                        start: float, seconds: float, title: str, path: Path) -> None:
    """One excerpt: breathing strength with both 'normal level' trackers, and the events each one finds."""
    m = (envelope["time"] >= start) & (envelope["time"] < start + seconds)
    minutes = (envelope.loc[m, "time"] - start) / 60
    flow = envelope["envelope"].rolling(5, center=True, min_periods=1).mean()[m] / night_median
    fig, axes = plt.subplots(2, 1, figsize=(14, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.fill_between(minutes, 0, flow.clip(upper=3), color=FLOW_FILL, alpha=0.8, linewidth=0, label="breathing strength (5 s mean)")
    for name, values in baselines.items():
        color, style = BASELINE_STYLE.get(name, ("black", ":"))
        ax.plot(minutes, values[m.to_numpy()] / night_median, color=color, linestyle=style, linewidth=1.8, label=f"normal level: {name.replace('_', ' ')}")
    ax.set(ylabel="breathing vs typical night", ylim=(0, 3))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    strip = axes[1]
    for i, (name, rows) in enumerate(events.items()):
        rows = rows[(rows["end_time"] > start) & (rows["start_time"] < start + seconds)]
        for r in rows.itertuples():
            strip.barh(i, (min(r.end_time, start + seconds) - max(r.start_time, start)) / 60, left=(max(r.start_time, start) - start) / 60, height=0.6,
                       color=EVENT_COLORS.get(r.kind, "grey"), linewidth=0)
    strip.set_yticks(range(len(events)), [f"{n.replace('_', ' ')} ({len(r[(r['end_time'] > start) & (r['start_time'] < start + seconds)])})" for n, r in events.items()])
    strip.set_ylim(-0.6, len(events) - 0.4)
    strip.grid(False)
    strip.tick_params(axis="y", length=0)
    strip.set_xlim(0, seconds / 60)
    strip.set_xlabel(f"minutes from {int(start // 3600)}:{int(start % 3600 // 60):02d}:00")
    strip.legend(handles=[Patch(color=c, label=k) for k, c in EVENT_COLORS.items()], loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    _header(fig, title, "An event is breathing well below the 'normal level' line. The two lines estimate that level differently; the bars show what each finds.")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_supervised(frame: pd.DataFrame, probability: np.ndarray, auc: float | None, f1: float, title: str, path: Path) -> None:
    """Predicted event probability over the validation night with the rule-based event windows as a band, and the ROC curve."""
    from sklearn.metrics import roc_curve

    is_event = frame["event"].isin(["apnea", "hypopnea"]).to_numpy()
    start, end = frame["window_start_time"].to_numpy(float) / 3600, frame["window_end_time"].to_numpy(float) / 3600
    fig = plt.figure(figsize=(15, 4.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 0.5], width_ratios=[4, 1.25], hspace=0.12, wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    ax.fill_between(np.repeat(np.c_[start, end].ravel(), 1), 0, np.repeat(probability, 2), color=FLOW_FILL, linewidth=0)
    ax.plot(np.c_[start, end].ravel(), np.repeat(probability, 2), color=FLOW_COLOR, linewidth=0.5)
    ax.axhline(0.5, color="#444", linestyle=":", linewidth=0.8)
    ax.text(end[-1], 0.52, "decision line ", ha="right", va="bottom", fontsize=8, color="#444")
    ax.set(ylabel="model confidence\nthat a window is an event", ylim=(0, 1), xlim=(start[0], end[-1]))
    ax.tick_params(labelbottom=False)
    band = fig.add_subplot(grid[1, 0], sharex=ax)
    _band(band, start[is_event], end[is_event], np.array(["hypopnea"] * int(is_event.sum()), dtype=object), EVENT_COLORS)
    _band_axis(band, "events (rules)")
    band.set_xlabel("hours since the recording started")
    roc = fig.add_subplot(grid[:, 1])
    if auc is not None and is_event.any() and not is_event.all():
        fpr, tpr, _ = roc_curve(is_event, probability)
        roc.fill_between(fpr, 0, tpr, color="#deebf7", step="post", linewidth=0)
        roc.plot(fpr, tpr, color="#2c7fb8", linewidth=2, drawstyle="steps-post", label=f"area {auc:.2f} (1 = perfect)")
    roc.plot([0, 1], [0, 1], color="#999", linestyle=":", linewidth=0.8, label="guessing (0.5)")
    roc.set(title="Hits vs false alarms (ROC)", xlabel="false alarms (share of normal windows)", ylabel="events caught", xlim=(0, 1), ylim=(0, 1))
    roc.legend(loc="lower right")
    _header(fig, title, f"Where the blue area rises above the line the model calls an event; the orange band is what the rules found (F1 {f1:.2f}).")
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
            ax.set(title=f"{space} - {NIGHT.get(stage, stage)}", xlabel=Z.columns[0], ylabel=Z.columns[1])
            ax.legend(markerscale=1.8, loc="best")
    fig.tight_layout()
    _header(fig, title, "Each dot is a 30 s window, coloured by what the rules found in it. Grey = normal breathing.")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ SpO2 reference
OXIMETRY_COLORS = {"confirmed": "#2c7fb8", "explained": "#e6550d", "chance": "#9a9a9a", "spo2": "#1a1a1a"}


def save_oximetry_scores(scores: pd.DataFrame, path: Path) -> None:
    """One panel per stage, one row per method: chance -> confirmed share (lift) and share of desaturations explained."""
    stages = [s for s in ("training", "validation") if s in set(scores["stage"])]
    methods = list(dict.fromkeys(scores["method"]))
    fig, axes = plt.subplots(1, len(stages), figsize=(6.2 * len(stages), 0.42 * len(methods) + 1.6), sharey=True, squeeze=False)
    y = np.arange(len(methods))[::-1]
    for ax, stage in zip(axes[0], stages):
        rows = scores[scores["stage"] == stage].set_index("method").reindex(methods)
        ax.hlines(y, rows["chance"], rows["confirmed"], color=OXIMETRY_COLORS["confirmed"], linewidth=2, alpha=0.35)
        ax.scatter(rows["chance"], y, s=40, facecolors="white", edgecolors=OXIMETRY_COLORS["chance"], linewidths=1.5, label="followed by a dip by chance", zorder=3)
        ax.scatter(rows["confirmed"], y, s=48, color=OXIMETRY_COLORS["confirmed"], edgecolors="white", linewidths=1.5, label="events followed by an oxygen dip", zorder=4)
        ax.scatter(rows["explained"], y - 0.22, s=48, marker="D", color=OXIMETRY_COLORS["explained"], edgecolors="white", linewidths=1.5, label="oxygen dips explained", zorder=4)
        for yi, (c, lift) in zip(y, rows[["confirmed", "lift"]].to_numpy(float)):
            if np.isfinite(c) and np.isfinite(lift):
                ax.annotate(f"x{lift:.1f}", (c, yi), xytext=(6, 4), textcoords="offset points", fontsize=7, color="#444")
        ax.set(title=NIGHT.get(stage, stage), xlim=(0, 1), xlabel="share (during sleep)")
        ax.set_yticks(y, methods)
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, -0.12 - 0.3 / max(len(methods), 1)), ncol=3)
    fig.tight_layout()
    _header(fig, "Do the detected events show up in blood oxygen?",
            "Blue: share of each method's events followed by an oxygen dip; grey ring: the same by pure chance (x = how many times better). "
            "Orange: share of oxygen dips the method explains.")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_oximetry_excerpt(envelope: pd.DataFrame, events: pd.DataFrame, spo2: pd.DataFrame, desaturations: pd.DataFrame, night_median: float,
                          start: float, seconds: float, lag_seconds: float, title: str, path: Path) -> None:
    """Airflow envelope with events (solid = confirmed by a desaturation, hatched = not) above SpO2 with desaturation nadirs."""
    end = start + seconds
    fig, (ax, ox) = plt.subplots(2, 1, figsize=(14, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    m = (envelope["time"] >= start) & (envelope["time"] < end)
    flow = envelope["envelope"].rolling(3, center=True, min_periods=1).mean()[m] / night_median  # 3 s mean: one hump per breath
    ax.fill_between((envelope.loc[m, "time"] - start) / 60, 0, flow, color="#9ecae1", alpha=0.8, linewidth=0)
    rows = events[(events["end_time"] > start) & (events["start_time"] < end) & events["kind"].isin(("apnea", "hypopnea"))]
    for r in rows.itertuples():
        linked = np.isfinite(getattr(r, "desaturation", np.nan))
        ax.axvspan((r.start_time - start) / 60, (r.end_time - start) / 60, color=EVENT_COLORS[r.kind], alpha=0.45 if linked else 0.12,
                   hatch=None if linked else "///", linewidth=0)
    ax.set(ylabel="breathing vs typical night", ylim=(0, 2.5))
    ax.legend(handles=[Patch(color="#9ecae1", label="breathing strength (3 s mean)"),
                       *[Patch(color=c, alpha=0.45, label=f"{k}, confirmed") for k, c in EVENT_COLORS.items() if k != "artefact"],
                       Patch(facecolor="white", edgecolor="#999", hatch="///", label="not confirmed")],
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    m = (spo2["time"] >= start) & (spo2["time"] < end)
    ox.plot((spo2.loc[m, "time"] - start) / 60, spo2.loc[m, "spo2"], color=OXIMETRY_COLORS["spo2"], linewidth=1.1, label="SpO2 (1 s median)")
    d = desaturations[(desaturations["nadir_time"] >= start) & (desaturations["nadir_time"] < end)]
    nadir = spo2.set_index("time")["spo2"].reindex(d["nadir_time"]).to_numpy()
    ox.scatter((d["nadir_time"] - start) / 60, nadir, s=36, marker="v", color=OXIMETRY_COLORS["explained"], edgecolors="white", linewidths=1, zorder=3,
               label=f"desaturation nadir (confirms an event ending up to {lag_seconds:.0f} s earlier)")
    ox.set(ylabel="blood oxygen SpO2 (%)", xlim=(0, seconds / 60), xlabel=f"minutes from {int(start // 3600)}:{int(start % 3600 // 60):02d}:00")
    ox.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    _header(fig, title, "Each breathing event (shaded) is followed about 20 seconds later by a dip in blood oxygen (triangles): the delay of blood travelling to the finger.")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ algorithm cards (one layout for every detector)
NORMAL_COLOR = "#c7c7c7"


def save_card(card: dict, path: Path) -> None:
    """One figure per algorithm, same layout for all.

    Top: an excerpt of breathing strength with the algorithm's decision as a band underneath. Middle: its events next to
    the rule events. Bottom left: what each of its groups / states looks like over the flow ratio. Bottom right: an
    annotated matrix (group contents, transition probabilities, or agreement with the rules).
    """
    ex, dens_cfg, mat_cfg = card["excerpt"], card["density"], card["matrix"]
    fig = plt.figure(figsize=(15, 8.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[2.2, 0.9, 2.4], width_ratios=[1.6, 1], hspace=0.5, wspace=0.28)
    ax = fig.add_subplot(grid[0, :])
    start = ex["time"][0]
    minutes = (ex["time"] - start) / 60
    ax.fill_between(minutes, 0, np.clip(ex["ratio"], 0, 3), color=FLOW_FILL, alpha=0.8, linewidth=0)
    b_start, b_end, b_labels = ex["band"]
    _band(ax, (np.asarray(b_start) - start) / 60, (np.asarray(b_end) - start) / 60, b_labels, ex["band_colors"], y=-0.25, height=0.2)
    ax.axhline(0, color="#444", linewidth=0.6)
    ax.set(ylabel="breathing vs normal level", ylim=(-0.3, 3), xlim=(0, minutes[-1]))
    ax.legend(handles=[Patch(color=FLOW_FILL, label="breathing strength (5 s mean)")] +
              [Patch(color=ex["band_colors"][k], label=text) for k, text in ex["band_names"].items()],
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, title=ex.get("band_title"), title_fontsize=8)
    strip = fig.add_subplot(grid[1, :], sharex=ax)
    strip.grid(False)
    strip.tick_params(axis="y", length=0)
    for i, (name, rows) in enumerate(ex["events"].items()):
        for r in rows.itertuples():
            strip.barh(i, (r.end_time - r.start_time) / 60, left=(r.start_time - start) / 60, height=0.6, color=EVENT_COLORS.get(r.kind, EVENT_COLORS["hypopnea"]), linewidth=0)
    strip.set_yticks(range(len(ex["events"])), list(ex["events"]))
    strip.set_ylim(-0.6, len(ex["events"]) - 0.4)
    strip.set_xlabel(f"minutes from {int(start // 3600)}:{int(start % 3600 // 60):02d}:00 (night 2)")
    strip.legend(handles=[Patch(color=EVENT_COLORS[k], label=k) for k in ("apnea", "hypopnea")], loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    dens = fig.add_subplot(grid[2, 0])
    y = dens_cfg["y"]
    lo, hi = np.log(0.03), np.log(5)
    dens.hist(np.clip(y, lo, hi), bins=np.linspace(lo, hi, 120), density=True, color="#e0e0e0", linewidth=0, label="night 1, all seconds")
    for label, color, gy, pdf in dens_cfg["curves"]:
        dens.plot(gy, pdf, color=color if color != NORMAL_COLOR else "#7f7f7f", linewidth=2, label=label)
    ticks = [0.05, 0.1, 0.3, 0.7, 1, 2, 4]
    dens.set_xticks(np.log(ticks), [f"{t:g}" for t in ticks])
    dens.axvline(np.log(0.7), color="#999", linestyle=":", linewidth=0.8)
    dens.set(title=dens_cfg.get("title", "What each group looks like (night 1)"), xlabel="breathing vs normal level (log scale; dotted = 0.7 event line)",
             ylabel="density", xlim=(lo, hi))
    dens.legend(loc="upper left", fontsize=7.5)

    mat = fig.add_subplot(grid[2, 1])
    mat.grid(False)
    values = np.asarray(mat_cfg["values"], float)
    vmax = mat_cfg.get("vmax", 1.0)
    mat.imshow(values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    fmt = mat_cfg.get("fmt", lambda v: f"{v:.0%}")
    extra = mat_cfg.get("extra", {})
    for (i, j), v in np.ndenumerate(values):
        text = "-" if not np.isfinite(v) else fmt(v) + extra.get((i, j), "")
        mat.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if np.isfinite(v) and v > 0.6 * vmax else "#1a1a1a")
    mat.set_xticks(range(values.shape[1]), mat_cfg["cols"], fontsize=8)
    mat.set_yticks(range(values.shape[0]), mat_cfg["rows"], fontsize=8)
    mat.tick_params(length=0)
    mat.set(title=mat_cfg["title"], xlabel=mat_cfg.get("xlabel", ""), ylabel=mat_cfg.get("ylabel", ""))
    _header(fig, card["title"], card.get("subtitle"))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
