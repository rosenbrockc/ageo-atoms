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

def witness_show(io: AbstractArray, s: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness__zero_offset(seconds: AbstractArray) -> AbstractArray:
    """Ghost witness for  Zero Offset."""
    result = AbstractArray(
        shape=seconds.shape,
        dtype="float64",
    )
    return result

def witness_apply_offsets(sec: AbstractArray, ts1: AbstractArray, ts2: AbstractArray) -> AbstractArray:
    """Ghost witness for Apply Offsets."""
    result = AbstractArray(
        shape=sec.shape,
        dtype="float64",
    )
    return result
