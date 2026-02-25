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

def witness_velocitystatereadout(state_in: AbstractArray) -> AbstractArray:
    """Ghost witness for VelocityStateReadout."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_posequeryaccessors() -> AbstractArray:
    """Ghost witness for PoseQueryAccessors."""
    return None
