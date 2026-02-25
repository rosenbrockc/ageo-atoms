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

def witness_initializebacklashfilterstate() -> AbstractArray:
    """Ghost witness for InitializeBacklashFilterState."""
    return None

def witness_updatealphaparameter(state_in: AbstractArray, alpha_in: AbstractArray) -> AbstractArray:
    """Ghost witness for UpdateAlphaParameter."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_updatecrossingtimemaximum(state_in: AbstractArray, t_crossing_max_in: AbstractArray) -> AbstractArray:
    """Ghost witness for UpdateCrossingTimeMaximum."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result
