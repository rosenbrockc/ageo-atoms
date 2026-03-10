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

def witness_cubicsplinefit(x: AbstractArray, y: AbstractArray, axis: AbstractArray, bc_type: AbstractArray, extrapolate: AbstractArray) -> AbstractArray:
    """Ghost witness for CubicSplineFit."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_rbfinterpolatorfit(y: AbstractArray, d: AbstractArray, neighbors: AbstractArray, smoothing: AbstractArray, kernel: AbstractArray, epsilon: AbstractArray, degree: AbstractArray) -> AbstractArray:
    """Ghost witness for RBFInterpolatorFit."""
    result = AbstractArray(
        shape=y.shape,
        dtype="float64",
    )
    return result
