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

def witness_initialize_glft_state() -> AbstractArray:
    """Ghost witness for initialize_glft_state."""
    return None

def witness_update_glft_coefficients(last_c1: AbstractArray, last_c2: AbstractArray, xi: AbstractArray, gamma: AbstractArray, delta: AbstractArray, A: AbstractArray, k: AbstractArray) -> AbstractArray:
    """Ghost witness for update_glft_coefficients."""
    result = AbstractArray(
        shape=last_c1.shape,
        dtype="float64",
    )
    return result

def witness_evaluate_spread_conditions(c1: AbstractArray, c2: AbstractArray, delta: AbstractArray, volatility: AbstractArray, adj1: AbstractArray, threshold: AbstractArray) -> AbstractArray:
    """Ghost witness for evaluate_spread_conditions."""
    result = AbstractArray(
        shape=c1.shape,
        dtype="float64",
    )
    return result
