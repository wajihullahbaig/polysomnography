"""Local-level Kalman filter and RTS smoother for 1-D amplitude trends.

Model: level_t = level_{t-1} + w_t (process noise q), observation y_t = level_t + v_t (noise r).
The smoothed level is a slow, two-sided estimate of the breathing amplitude: with a small
`process_ratio` (= q / r) the level ignores individual events and tracks the baseline, so
airflow events show up as windows where the envelope falls well below the Kalman level.
"""

from __future__ import annotations

import numpy as np


def kalman_smooth(y: np.ndarray, process_ratio: float = 1e-5, measurement_var: float | None = None) -> np.ndarray:
    """Rauch-Tung-Striebel smoothed local level of `y`.

    `measurement_var` defaults to the variance of `y` (breath-to-breath variation is the noise);
    the process variance is `process_ratio` times that. Steady-state gain ~ sqrt(process_ratio), so
    1e-5 gives a time constant of a few hundred samples (minutes at 1 Hz).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    r = float(np.var(y)) if measurement_var is None else float(measurement_var)
    r = max(r, 1e-12)
    q = max(process_ratio, 1e-12) * r

    level = np.empty(n)
    var = np.empty(n)
    pred_level = np.empty(n)
    pred_var = np.empty(n)
    x, p = y[0], r  # initial state
    for t in range(n):
        pred_level[t], pred_var[t] = x, p + q  # predict
        k = pred_var[t] / (pred_var[t] + r)  # update
        x = pred_level[t] + k * (y[t] - pred_level[t])
        p = (1 - k) * pred_var[t]
        level[t], var[t] = x, p

    smooth = level.copy()
    for t in range(n - 2, -1, -1):  # backward RTS pass
        c = var[t] / pred_var[t + 1]
        smooth[t] = level[t] + c * (smooth[t + 1] - pred_level[t + 1])
    return smooth

