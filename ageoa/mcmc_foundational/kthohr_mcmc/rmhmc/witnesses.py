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

def witness_buildrmhmctransitionkernel(target_log_kernel: AbstractArray, tensor_fn: AbstractArray) -> AbstractArray:
    """Ghost witness for BuildRMHMCTransitionKernel."""
    result = AbstractArray(
        shape=target_log_kernel.shape,
        dtype="float64",
    )
    return result
