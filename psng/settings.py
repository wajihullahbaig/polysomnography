"""Typed, validated settings loaded from YAML."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Stage = Literal["training", "validation"]


class Segment(BaseModel):
    file_path: Path
    start_seconds: float = Field(0.0, ge=0)
    end_seconds: float | None = Field(None, gt=0)  # None = until the end of the recording


class DataSettings(BaseModel):
    channel: str
    window_seconds: float = Field(gt=0)
    hop_seconds: float = Field(0.0, ge=0)  # 0 = non-overlapping windows
    smoothing_seconds: float = Field(0.5, ge=0)
    training: Segment
    validation: Segment


class SelectionSettings(BaseModel):
    type: Literal["none", "pca", "correlation"] = "none"
    pca_loading: float = Field(0.5, gt=0, le=1)  # keep features with |loading| above this
    pca_variance: float = Field(0.9, gt=0, le=1)  # on the components explaining this much variance
    correlation_method: Literal["pearson", "spearman", "kendall"] = "pearson"
    correlation_threshold: float = Field(0.9, gt=0, le=1)


class ReductionSettings(BaseModel):
    type: Literal["none", "pca"] = "none"
    n_components: int = Field(3, ge=1)


class FeatureSettings(BaseModel):
    extractor: str
    scale: bool = True
    selection: SelectionSettings = Field(default_factory=SelectionSettings)
    reduction: ReductionSettings = Field(default_factory=ReductionSettings)

    @field_validator("extractor")
    @classmethod
    def _known_extractor(cls, name: str) -> str:
        from psng.features import EXTRACTORS

        if name not in EXTRACTORS:
            raise ValueError(f"unknown extractor {name!r}; choose from {sorted(EXTRACTORS)}")
        return name


class RunSettings(BaseModel):
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    resume: bool = True
    n_jobs: int = -1
    chunk_windows: int = Field(64, ge=1)
    interactive_plots: bool = True
    plot_max_points: int = Field(100_000, ge=1000)


class Settings(BaseModel):
    data: DataSettings
    features: FeatureSettings
    models: dict[str, dict[str, Any]] = Field(min_length=1)
    run: RunSettings = Field(default_factory=RunSettings)

    @field_validator("models", mode="before")
    @classmethod
    def _known_models(cls, models: dict[str, Any]) -> dict[str, dict[str, Any]]:
        from psng.clustering import CLUSTERERS

        unknown = set(models) - set(CLUSTERERS)
        if unknown:
            raise ValueError(f"unknown clusterers {sorted(unknown)}; choose from {sorted(CLUSTERERS)}")
        return {name: dict(params or {}) for name, params in models.items()}

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def segment(self, stage: Stage) -> Segment:
        return getattr(self.data, stage)

    def feature_key(self, stage: Stage) -> str:
        """Hash of everything that determines the extracted features of a stage."""
        payload = {
            "segment": self.segment(stage).model_dump(mode="json"),
            "data": self.data.model_dump(mode="json", exclude={"training", "validation"}),
            "extractor": self.features.extractor,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
