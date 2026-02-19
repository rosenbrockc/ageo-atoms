from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field

class PPGState(BaseModel):
    """State for windowed PPG processing and reconstruction."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sampling_rate: int | None = Field(default=None)
    buffer: list[float] | None = Field(default=None)
    is_reliable: bool | None = Field(default=None)
    clean_indices: list[list[int]] | None = Field(default=None)
    noisy_indices: list[list[int]] | None = Field(default=None)
