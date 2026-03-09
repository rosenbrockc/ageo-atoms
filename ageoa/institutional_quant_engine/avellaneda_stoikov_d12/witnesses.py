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

def witness_marketmakerstateinit(s0: AbstractArray, inventory: AbstractArray) -> AbstractArray:
    """Ghost witness for MarketMakerStateInit."""
    result = AbstractArray(
        shape=s0.shape,
        dtype="float64",
    )
    return result

def witness_optimalquotecalculation(gamma: AbstractArray, k: AbstractArray, q: AbstractArray, s: AbstractArray, sigma: AbstractArray) -> AbstractArray:
    """Ghost witness for OptimalQuoteCalculation."""
    result = AbstractArray(
        shape=gamma.shape,
        dtype="float64",
    )
    return result
