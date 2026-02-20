"""Pydantic state model for legacy ECG ingestion artifacts."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ECGPipelineState(BaseModel):
    """Intermediate pipeline state carrying the band-filtered signal and detected peak indices between processing stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filtered: np.ndarray | None = Field(default=None)
    rpeaks: np.ndarray | None = Field(default=None)
