"""Hidden Markov model breathing-event detector: a non-clustering alternative to fixed flow thresholds.

The 1 Hz airflow envelope relative to its local baseline is smoothed over about one breath and log-transformed,
y_t = log(ratio_t). A Gaussian HMM with K states explains it: each state k emits y ~ N(mu_k, sigma_k^2) and the
state sequence is a Markov chain with transition matrix A. Unlike clustering, every second is judged together
with its neighbours: the learned "stay" probabilities make a state persist, so one deeper breath inside a
hypopnea does not end the event.

Fitting (night 1 only) is Baum-Welch (expectation-maximisation with a scaled forward-backward pass); decoding is
Viterbi, the single most likely state sequence. States are ordered by mean flow and named from it; a state is an
*event state* when its mean flow is below the hypopnea line (`events.hypopnea_ratio`, AASM 30 % drop), so which
states count as events follows the AASM rule rather than any tuning. Runs of event states become events with the
same duration and apnea / artefact rules as the threshold detector.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from psng.events import EVENT_COLUMNS, _runs, local_baseline
from psng.settings import EventSettings, HmmSettings

RATIO_FLOOR = 0.02  # log of a flat line would be -inf


def hmm_observation(envelope: np.ndarray, events: EventSettings, smoothing_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """(ratio, y): flow / local baseline per second, and its log after a centred moving mean over `smoothing_seconds`."""
    ratio = envelope / np.maximum(local_baseline(envelope, events), 1e-12)
    smooth = pd.Series(ratio).rolling(max(1, int(smoothing_seconds)), center=True, min_periods=1).mean().to_numpy()
    return ratio, np.log(np.clip(smooth, RATIO_FLOOR, None))


class GaussianHMM:
    """K-state HMM with one Gaussian per state, fitted by Baum-Welch and decoded by Viterbi."""

    def __init__(self, n_states: int = 3, max_iter: int = 100, tol: float = 1e-4, stay: float = 0.95):
        self.n_states, self.max_iter, self.tol, self.stay = n_states, max_iter, tol, stay
        self.names_: list[str] = []
        self.event_states_: list[int] = []

    # ---------------------------------------------------------------- model
    def _emissions(self, y: np.ndarray) -> np.ndarray:
        """b[t, k] = N(y_t; mu_k, var_k)."""
        return np.exp(-0.5 * (y[:, None] - self.means_) ** 2 / self.vars_) / np.sqrt(2 * np.pi * self.vars_) + 1e-300

    def _forward_backward(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Scaled forward-backward: state posteriors gamma[t, k], expected transition counts xi[j, k] and the log-likelihood."""
        b, A, n = self._emissions(y), self.transitions_, len(y)
        alpha, beta, c = np.empty((n, self.n_states)), np.ones((n, self.n_states)), np.empty(n)
        alpha[0] = self.start_ * b[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * b[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        for t in range(n - 2, -1, -1):
            beta[t] = A @ (b[t + 1] * beta[t + 1]) / c[t + 1]
        gamma = alpha * beta
        xi = A * (alpha[:-1].T @ (b[1:] * beta[1:] / c[1:, None]))
        return gamma, xi, float(np.log(c).sum())

    def fit(self, y: np.ndarray) -> GaussianHMM:
        y = np.asarray(y, float)
        k = self.n_states
        self.means_ = np.quantile(y, np.linspace(0.1, 0.9, k))  # spread over the observed range
        self.vars_ = np.full(k, y.var() / k)
        self.transitions_ = np.full((k, k), (1 - self.stay) / max(k - 1, 1))
        np.fill_diagonal(self.transitions_, self.stay if k > 1 else 1.0)
        self.start_ = np.full(k, 1 / k)
        self.log_likelihood_ = -np.inf
        for self.n_iter_ in range(1, self.max_iter + 1):
            gamma, xi, ll = self._forward_backward(y)
            weight = gamma.sum(axis=0)
            self.start_ = (gamma[0] + 1e-6) / (gamma[0] + 1e-6).sum()
            self.transitions_ = xi / xi.sum(axis=1, keepdims=True)
            self.means_ = gamma.T @ y / weight
            self.vars_ = np.maximum((gamma * (y[:, None] - self.means_) ** 2).sum(axis=0) / weight, 1e-4)
            done = ll - self.log_likelihood_ < self.tol * len(y)
            self.log_likelihood_ = ll
            if done:
                break
        order = np.argsort(self.means_)  # state 0 = lowest flow
        self.means_, self.vars_, self.start_ = self.means_[order], self.vars_[order], self.start_[order]
        self.transitions_ = self.transitions_[np.ix_(order, order)]
        return self

    def decode(self, y: np.ndarray) -> np.ndarray:
        """Viterbi: the most likely state sequence."""
        logb, logA = np.log(self._emissions(np.asarray(y, float))), np.log(self.transitions_ + 1e-300)
        n = len(logb)
        back = np.empty((n, self.n_states), dtype=int)
        score = np.log(self.start_) + logb[0]
        for t in range(1, n):
            step = score[:, None] + logA
            back[t] = step.argmax(axis=0)
            score = step.max(axis=0) + logb[t]
        path = np.empty(n, dtype=int)
        path[-1] = score.argmax()
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def posterior(self, y: np.ndarray) -> np.ndarray:
        return self._forward_backward(np.asarray(y, float))[0]

    # ---------------------------------------------------------------- names and persistence
    def name_states(self, events: EventSettings) -> GaussianHMM:
        """Name states by their mean flow ratio; event states are the ones below the hypopnea line."""
        flow = np.exp(self.means_)
        self.event_states_ = [k for k, f in enumerate(flow) if f < events.hypopnea_ratio]
        names = ["absent flow" if f < events.apnea_ratio else "reduced flow" if f < events.hypopnea_ratio else "large breaths" if f > 1.5
                 else "normal breathing" for f in flow]
        self.names_ = [n if names.count(n) == 1 else f"{n} #{names[: k + 1].count(n)}" for k, n in enumerate(names)]
        return self

    @property
    def mean_flow(self) -> np.ndarray:
        return np.exp(self.means_)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> GaussianHMM:
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_hmm(envelope: np.ndarray, events: EventSettings, cfg: HmmSettings) -> GaussianHMM:
    _, y = hmm_observation(envelope, events, cfg.smoothing_seconds)
    return GaussianHMM(cfg.n_states, cfg.max_iter, cfg.tol).fit(y).name_states(events)


def hmm_events(envelope: pd.DataFrame, model: GaussianHMM, events: EventSettings, cfg: HmmSettings) -> tuple[pd.DataFrame, np.ndarray]:
    """Events (EVENT_COLUMNS) from runs of event states, and the Viterbi state of every second."""
    env, time = envelope["envelope"].to_numpy(float), envelope["time"].to_numpy(float)
    if len(env) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS), np.array([], dtype=int)
    ratio, y = hmm_observation(env, events, cfg.smoothing_seconds)
    states = model.decode(y)
    rows = []
    for start, stop in _runs(np.isin(states, model.event_states_)):
        duration = float(stop - start)
        if duration < events.min_seconds:
            continue
        deepest = max((b - a for a, b in _runs(ratio[start:stop] < events.apnea_ratio)), default=0)
        kind = "artefact" if duration > events.max_seconds else "apnea" if deepest >= events.min_seconds else "hypopnea"
        rows.append((time[start], time[start] + duration, duration, kind, float(ratio[start:stop].min())))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS), states
