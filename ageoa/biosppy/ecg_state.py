"""Auto-generated Pydantic state models for cross-window state."""

from __future__ import annotations

from pydantic import BaseModel, Field

class ECGPipelineState(BaseModel):
    """Intermediate pipeline state carrying the filtered ECG signal and detected R-peak indices between stages"""
    filtered: np.ndarray = Field(default=None)
    rpeaks: np.ndarray = Field(default=None)
