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

def witness_shgoglobaloptimization(func: AbstractArray, bounds: AbstractArray, args: AbstractArray, constraints: AbstractArray, n: AbstractArray, iters: AbstractArray, callback: AbstractArray, minimizer_kwargs: AbstractArray, options: AbstractArray, sampling_method: AbstractArray) -> AbstractArray:
    """Ghost witness for ShgoGlobalOptimization."""
    result = AbstractArray(
        shape=func.shape,
        dtype="float64",
    )
    return result

def witness_differentialevolutionoptimization(func: AbstractArray, bounds: AbstractArray, args: AbstractArray, strategy: AbstractArray, maxiter: AbstractArray, popsize: AbstractArray, tol: AbstractArray, mutation: AbstractArray, recombination: AbstractArray, seed: AbstractArray, callback: AbstractArray, disp: AbstractArray, polish: AbstractArray, init: AbstractArray, atol: AbstractArray, updating: AbstractArray, workers: AbstractArray, constraints: AbstractArray, x0: AbstractArray) -> AbstractArray:
    """Ghost witness for DifferentialEvolutionOptimization."""
    result = AbstractArray(
        shape=func.shape,
        dtype="float64",
    )
    return result
