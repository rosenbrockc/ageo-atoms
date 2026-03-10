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

def witness_singlesourceshortestpath(csgraph: AbstractArray, directed: AbstractArray, indices: AbstractArray, return_predecessors: AbstractArray, unweighted: AbstractArray, limit: AbstractArray, min_only: AbstractArray) -> AbstractArray:
    """Ghost witness for SingleSourceShortestPath."""
    result = AbstractArray(
        shape=csgraph.shape,
        dtype="float64",
    )
    return result

def witness_allpairsshortestpath(csgraph: AbstractArray, directed: AbstractArray, return_predecessors: AbstractArray, unweighted: AbstractArray) -> AbstractArray:
    """Ghost witness for AllPairsShortestPath."""
    result = AbstractArray(
        shape=csgraph.shape,
        dtype="float64",
    )
    return result

def witness_minimumspanningtree(csgraph: AbstractArray, overwrite: AbstractArray) -> AbstractArray:
    """Ghost witness for MinimumSpanningTree."""
    result = AbstractArray(
        shape=csgraph.shape,
        dtype="float64",
    )
    return result
