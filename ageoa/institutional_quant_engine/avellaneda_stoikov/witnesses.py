"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_initializemarketmakerstate(s0: AbstractArray, inventory: AbstractArray) -> AbstractArray:
    """Ghost witness for InitializeMarketMakerState."""
    result = AbstractArray(
        shape=s0.shape,
        dtype="float64",
    )
    return result

def witness_computeinventoryadjustedquotes(state_model: AbstractArray) -> AbstractArray:
    """Ghost witness for ComputeInventoryAdjustedQuotes."""
    result = AbstractArray(
        shape=state_model.shape,
        dtype="float64",
    )
    return result
