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

def witness_nuts_recursive_tree_build(direction_val: AbstractArray, step_size: AbstractArray, log_slice_variable: AbstractArray, initial_hmc_state: AbstractArray, log_prob_oracle: AbstractArray, integrator_fn: AbstractArray, tree_depth: AbstractArray) -> AbstractArray:
    """Ghost witness for nuts_recursive_tree_build."""
    result = AbstractArray(
        shape=direction_val.shape,
        dtype="float64",
    )
    return result
