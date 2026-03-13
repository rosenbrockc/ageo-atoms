from __future__ import annotations
from typing import Any
"""Auto-generated Pydantic state models for cross-window state."""



import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

from pydantic import BaseModel, ConfigDict, Field

class OrderState(BaseModel):
    """Represents the state of a single order in a queue, including its remaining quantity, the volume of orders ahead, and its fill status."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_filled: bool | None = Field(default=None)
    my_qty: float | None = Field(default=None)
    orders_ahead: float | None = Field(default=None)