"""Supervised reference: gradient boosting trained on the rule-based event labels of the training night.

No human scoring exists, so the classifier learns the detector's labels from the window features and
is tested on the other night. It answers "how much of the event information do the window features
carry?" and provides feature importances; it is an upper reference for the unsupervised clusterers,
not a clinical model. Uses XGBoost when installed, otherwise scikit-learn's histogram gradient boosting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from psng.data import feature_matrix
from psng.plots import save_supervised

try:  # optional dependency
    from xgboost import XGBClassifier

    BACKEND = "xgboost"
except ImportError:  # pragma: no cover - depends on the environment
    XGBClassifier = None
    BACKEND = "sklearn-histgb"

NON_FEATURES = ("cluster_id", "cluster_label")


class SupervisedResult(BaseModel):
    backend: str
    target: str
    n_train: int
    n_validation: int
    train_f1: float
    validation_f1: float
    validation_precision: float
    validation_recall: float
    validation_auc: float | None = None
    importances: dict[str, float] = {}


def make_classifier(positive_rate: float, random_state: int = 42):
    weight = (1 - positive_rate) / max(positive_rate, 1e-6)
    if XGBClassifier is not None:
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=weight, random_state=random_state, n_jobs=1, eval_metric="logloss")
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=random_state)


def event_classifier(train: pd.DataFrame, validation: pd.DataFrame, target: str = "event") -> tuple[SupervisedResult, np.ndarray]:
    """Fit on `train` windows (features + `event` labels), score on `validation`; positives are apnea / hypopnea windows.

    Returns the scores and the predicted event probability of every validation window.
    """
    Xt, Xv = _features(train), _features(validation)
    yt, yv = _target(train, target), _target(validation, target)
    model = make_classifier(float(yt.mean()))
    model.fit(Xt, yt)
    pt, pv = model.predict(Xt), model.predict(Xv)
    prob = model.predict_proba(Xv)[:, 1] if hasattr(model, "predict_proba") else None
    importances = getattr(model, "feature_importances_", None)
    return SupervisedResult(
        backend=BACKEND, target=target, n_train=len(train), n_validation=len(validation),
        train_f1=float(f1_score(yt, pt, zero_division=0)), validation_f1=float(f1_score(yv, pv, zero_division=0)),
        validation_precision=float(precision_score(yv, pv, zero_division=0)), validation_recall=float(recall_score(yv, pv, zero_division=0)),
        validation_auc=float(roc_auc_score(yv, prob)) if prob is not None and len(np.unique(yv)) > 1 else None,
        importances={} if importances is None else dict(sorted(zip(Xt.columns, map(float, importances)), key=lambda kv: -kv[1])),
    ), (pv.astype(float) if prob is None else prob)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return feature_matrix(frame).drop(columns=[c for c in NON_FEATURES if c in frame.columns])


def _target(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target != "event":
        raise ValueError(f"unknown target {target!r}")
    return frame["event"].isin(["apnea", "hypopnea"]).to_numpy(int)


def run_supervised(output_dir: Path, model_csv: str) -> SupervisedResult | None:
    """Train on <output_dir>/training/csvs/<model_csv>, test on the validation one; None when a stage is missing."""
    paths = [Path(output_dir) / stage / "csvs" / model_csv for stage in ("training", "validation")]
    if not all(p.exists() for p in paths):
        return None
    train, validation = (pd.read_csv(p) for p in paths)
    result, probability = event_classifier(train, validation)
    comparison = Path(output_dir) / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    (comparison / "supervised.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    save_supervised(validation, probability, result.validation_auc, result.validation_f1,
                    "XGBoost on night 2: how sure the model is that each 30 s window holds an event", comparison / "xgboost.png")
    return result
