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

def witness_encodedistancematrix(mat_list: AbstractArray, max_cdr3: AbstractArray, max_epi: AbstractArray) -> AbstractArray:
    """Ghost witness for EncodeDistanceMatrix."""
    result = AbstractArray(
        shape=mat_list.shape,
        dtype="float64",
    )
    return result
