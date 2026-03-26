from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field


class MINTProcessingState(BaseModel):
    """State for composite categorical sequence processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: torch.Tensor | None = Field(default=None)
    chain_ids: torch.Tensor | None = Field(default=None)
    representations: torch.Tensor | None = Field(default=None)

    # Deterministic randomness control for stochastic transformer stubs.
    rng_seed: int = Field(default=0)
    rng_counter: int = Field(default=0, ge=0)
