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

def witness_buildnutstree(rng: AbstractArray, hamiltonian: AbstractArray, start_state: AbstractArray, direction: AbstractArray, tree_depth: AbstractArray, initial_energy: AbstractArray) -> AbstractArray:
    """Ghost witness for BuildNutsTree."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_nutstransitionkernel(rng: AbstractArray, hamiltonian: AbstractArray, initial_state: AbstractArray, trajectory_params: AbstractArray) -> AbstractArray:
    """Ghost witness for NutsTransitionKernel."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result
