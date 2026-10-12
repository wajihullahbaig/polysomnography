"""Static (PNG) and interactive (HTML) cluster plots written to disk; nothing pops up."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.colors import qualitative  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from psng.data import SignalData  # noqa: E402


def save_scatter(X: np.ndarray, labels: np.ndarray, centers: np.ndarray | None, title: str, path: Path) -> None:
    """2-D PCA projection of windows coloured by cluster."""
    if X.shape[1] < 2:
        X = np.c_[X, np.zeros(len(X))]
        centers = None if centers is None else np.c_[centers, np.zeros(len(centers))]
    pca = PCA(n_components=2).fit(X)
    points = pca.transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in np.unique(labels):
        m = labels == k
        ax.scatter(points[m, 0], points[m, 1], s=12, alpha=0.6, label="noise" if k < 0 else f"cluster {k}")
    if centers is not None and len(centers):
        c = pca.transform(centers)
        ax.scatter(c[:, 0], c[:, 1], c="black", marker="x", s=150, linewidths=3, label="centers")
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_interactive(signal: SignalData, frame: pd.DataFrame, title: str, path: Path, max_points: int = 100_000) -> None:
    """Signal coloured by cluster (one toggleable trace per cluster) plus windows-per-cluster bars.

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
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], vertical_spacing=0.12,
                        subplot_titles=("Clustered signal", "Windows per cluster"))
    fig.add_trace(go.Scattergl(x=t, y=y, name="signal", line=dict(color="lightgray", width=1), hoverinfo="skip"), row=1, col=1)
    for color, k in zip(colors, ids):
        idx = np.flatnonzero(cluster_of_sample == k)
        breaks = np.flatnonzero(np.diff(idx) > 1) + 1  # NaN separators between non-adjacent windows
        fig.add_trace(go.Scattergl(x=np.insert(t[idx], breaks, np.nan), y=np.insert(y[idx], breaks, np.nan),
                                   name=f"cluster {k}", line=dict(color=color, width=1.5)), row=1, col=1)
    counts = frame["cluster_id"].value_counts().sort_index()
    fig.add_trace(go.Bar(x=[f"cluster {k}" for k in counts.index], y=counts.values, marker_color=colors,
                         text=counts.values, textposition="auto", showlegend=False), row=2, col=1)

    n_traces = len(fig.data)
    buttons = [dict(label="All", method="restyle", args=[{"visible": [True] * n_traces}])]
    for i, k in enumerate(ids):
        visible = [j in (0, i + 1, n_traces - 1) for j in range(n_traces)]
        buttons.append(dict(label=f"Cluster {k}", method="restyle", args=[{"visible": visible}]))
    fig.update_layout(title=title, height=850, dragmode="pan",
                      updatemenus=[dict(type="buttons", direction="right", buttons=buttons, x=0, y=1.08, xanchor="left")])
    fig.update_xaxes(title_text="Time (s)", rangeslider=dict(visible=True, thickness=0.05), row=1, col=1)
    fig.update_yaxes(title_text=signal.channel, row=1, col=1)
    fig.write_html(path, include_plotlyjs=True)
