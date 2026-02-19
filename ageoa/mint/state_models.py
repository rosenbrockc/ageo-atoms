from __future__ import annotations
import torch
from pydantic import BaseModel, ConfigDict, Field

class MINTProcessingState(BaseModel):
    """State for multimeric protein sequence processing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: torch.Tensor | None = Field(default=None)
    chain_ids: torch.Tensor | None = Field(default=None)
    representations: torch.Tensor | None = Field(default=None)
