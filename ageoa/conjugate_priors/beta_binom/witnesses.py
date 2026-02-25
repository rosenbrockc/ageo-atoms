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

def witness_posterior_randmodel(pri: AbstractArray, G: AbstractArray, data: AbstractArray) -> AbstractArray:
    """Ghost witness for Posterior Randmodel."""
    result = AbstractArray(
        shape=pri.shape,
        dtype="float64",
    )
    return result

def witness_posterior_randmodel(pri: AbstractArray, G: AbstractArray, data: AbstractArray, w: AbstractArray) -> AbstractArray:
    """Ghost witness for Posterior Randmodel."""
    result = AbstractArray(
        shape=pri.shape,
        dtype="float64",
    )
    return result
