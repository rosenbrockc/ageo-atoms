"""Auto-generated Pydantic state models for cross-window state."""

from __future__ import annotations

from typing import Any

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

from pydantic import BaseModel, ConfigDict, Field

class ECGPipelineState(BaseModel):
    """Intermediate pipeline state carrying the filtered ECG signal and detected R-peak indices between stages"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    filtered: np.ndarray | None = Field(default=None)
    rpeaks: np.ndarray | None = Field(default=None)
