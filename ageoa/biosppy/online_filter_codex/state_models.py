"""Auto-generated Pydantic state models for cross-window state."""

from __future__ import annotations

from typing import Any

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

from pydantic import BaseModel, ConfigDict, Field

class FilterParamState(BaseModel):
    """Holds the normalized IIR/FIR filter coefficients produced by FilterStateInit and consumed read-only by every FilterStep call. `b` is the numerator coefficient array (len >= 1); `a` is the denominator coefficient array (len >= 1, a[0] != 0, normalized so a[0] == 1). These fields are immutable after initialization: no FilterStep transition may alter them. The filter order is implicitly max(len(a), len(b)) - 1."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    b: np.ndarray | None = Field(default=None)
    a: np.ndarray | None = Field(default=None)

class FilterDelayState(BaseModel):
    """Holds the IIR/FIR delay-line state vector that is threaded immutably through successive FilterStep calls. `zi` has shape (max(len(a), len(b)) - 1,) and is initialized to all-zeros by FilterStateInit.reset(). Invariant: each FilterStep atom consumes a `FilterDelayState` as state-in and produces a DISTINCT `FilterDelayState` as state-out, never mutating the input instance. This pure threading ensures correct online (sample-by-sample or block) filtering across arbitrary window boundaries."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    zi: np.ndarray | None = Field(default=None)
