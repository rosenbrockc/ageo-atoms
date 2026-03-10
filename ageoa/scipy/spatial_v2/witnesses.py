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

def witness_voronoitessellation(points: AbstractArray, incremental: AbstractArray, qhull_options: AbstractArray) -> AbstractArray:
    """Ghost witness for VoronoiTessellation."""
    result = AbstractArray(
        shape=points.shape,
        dtype="float64",
    )
    return result

def witness_delaunaytriangulation(points: AbstractArray, incremental: AbstractArray, qhull_options: AbstractArray) -> AbstractArray:
    """Ghost witness for DelaunayTriangulation."""
    result = AbstractArray(
        shape=points.shape,
        dtype="float64",
    )
    return result
