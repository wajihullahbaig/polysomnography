"""Band powers, band-limited amplitudes, spectral edge and spectral entropy."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import filtfilt, firwin

from psng.features.base import FeatureExtractor, power_entropy


class BandpassFeatures(FeatureExtractor):
    BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 100)}

    def __init__(self, sfreq: float):
        super().__init__(sfreq)
        if sfreq <= 60:
            raise ValueError("bandpass features need a sampling rate above 60 Hz (30 Hz high-pass)")
        self.taps = {  # FIR designs computed once, not per window
            "low_pass_amp": firwin(101, 4, fs=sfreq),
            "med_pass_amp": firwin(51, [4, 30], pass_zero=False, fs=sfreq),
            "high_pass_amp": firwin(25, 30, pass_zero=False, fs=sfreq),
        }

    def extract(self, window: np.ndarray) -> dict[str, float]:
        spectrum = np.abs(rfft(window))
        power = spectrum**2
        freqs = rfftfreq(len(window), 1 / self.sfreq)
        out = {f"{band}_power": float(power[(freqs >= lo) & (freqs <= hi)].sum()) for band, (lo, hi) in self.BANDS.items()}
        for name, taps in self.taps.items():
            out[name] = float(np.abs(filtfilt(taps, [1.0], window)).max())
        out["spectral_edge"] = float(np.percentile(spectrum, 95))
        out["spectral_entropy"] = power_entropy(power)
        return out
