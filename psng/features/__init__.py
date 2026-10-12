"""Feature extractor registry: settings.yaml `features.extractor` names map here."""

from psng.features.base import FeatureExtractor
from psng.features.basic import BasicFeatures
from psng.features.freq_time_pattern import FreqTimePattern
from psng.features.hjorth import HjorthFeatures
from psng.features.respiratory import RespiratoryFeatures
from psng.features.wavelet import WaveletFeatures

EXTRACTORS: dict[str, type[FeatureExtractor]] = {
    "respiratory": RespiratoryFeatures,
    "freqtimepatt": FreqTimePattern,
    "basic": BasicFeatures,
    "hjorth": HjorthFeatures,
    "wavelet": WaveletFeatures,
}


def make_extractor(name: str, sfreq: float) -> FeatureExtractor:
    return EXTRACTORS[name](sfreq)
