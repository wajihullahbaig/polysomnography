"""Load one EDF channel segment without preloading the whole recording."""

from __future__ import annotations

from pathlib import Path

import mne

from psng.data import SignalData


def load_channel(path: str | Path, channel: str, start_seconds: float = 0.0, end_seconds: float | None = None) -> SignalData:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if channel not in raw.ch_names:
        raise ValueError(f"channel {channel!r} not in {path}; available: {raw.ch_names}")
    sfreq = float(raw.info["sfreq"])
    total = raw.n_times / sfreq
    start = min(max(start_seconds, 0.0), total)
    end = total if end_seconds is None else min(max(end_seconds, start), total)
    data = raw.get_data(picks=[channel], start=int(start * sfreq), stop=int(end * sfreq))[0]
    return SignalData(signal=data, sfreq=sfreq, start_time=start, channel=channel)
