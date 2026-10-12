"""Gaussian HMM detector on synthetic breathing with known events."""

from __future__ import annotations

import numpy as np

from psng.events import flow_envelope
from psng.hmm import GaussianHMM, fit_hmm, hmm_events
from psng.settings import EventSettings, HmmSettings
from tests.test_events import APNEAS, HYPOPNEA, signal_with_events


def test_baum_welch_recovers_two_sticky_states():
    rng = np.random.default_rng(0)
    states = np.repeat(np.tile([0, 1], 20), 50)  # 40 blocks of 50 s
    y = np.where(states == 0, -1.0, 0.0) + 0.2 * rng.standard_normal(len(states))
    model = GaussianHMM(2).fit(y)
    assert np.allclose(model.means_, [-1, 0], atol=0.05) and (np.diag(model.transitions_) > 0.95).all()
    assert (model.decode(y) == states).mean() > 0.99
    assert np.allclose(model.posterior(y).sum(axis=1), 1)


def test_hmm_finds_the_synthetic_events_and_names_states(tmp_path):
    envelope = flow_envelope(signal_with_events())
    events_cfg, cfg = EventSettings(), HmmSettings()
    model = fit_hmm(envelope["envelope"].to_numpy(float), events_cfg, cfg)
    assert np.all(np.diff(model.means_) > 0) and model.event_states_[0] == 0 and len(set(model.names_)) == model.n_states
    assert all(model.mean_flow[k] < events_cfg.hypopnea_ratio for k in model.event_states_) and model.mean_flow[-1] > events_cfg.hypopnea_ratio
    events, states = hmm_events(envelope, model, events_cfg, cfg)
    assert len(states) == len(envelope)
    for start, end in [*APNEAS, HYPOPNEA]:  # every planted event is covered by a detected one
        assert ((events["start_time"] <= start + 5) & (events["end_time"] >= end - 5)).any()
    assert set(events["kind"]) <= {"apnea", "hypopnea"} and (events["kind"] == "apnea").sum() == 2
    model.save(tmp_path / "hmm.pkl")
    assert np.allclose(GaussianHMM.load(tmp_path / "hmm.pkl").transitions_, model.transitions_)
