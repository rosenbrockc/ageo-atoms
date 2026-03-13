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

def witness_iterate_pdb_atoms(element: AbstractArray) -> AbstractArray:
    """Ghost witness for iterate_pdb_atoms."""
    result = AbstractArray(
        shape=element.shape,
        dtype="float64",
    )
    return result

def witness_iterate_pdb_residues(element: AbstractArray) -> AbstractArray:
    """Ghost witness for iterate_pdb_residues."""
    result = AbstractArray(
        shape=element.shape,
        dtype="float64",
    )
    return result
