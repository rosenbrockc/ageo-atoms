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

def witness_binarysearchinsertion(a: AbstractArray, v: AbstractArray, side: AbstractArray, sorter: AbstractArray) -> AbstractArray:
    """Ghost witness for BinarySearchInsertion."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result

def witness_lexicographicindirectsort(keys: AbstractArray, axis: AbstractArray) -> AbstractArray:
    """Ghost witness for LexicographicIndirectSort."""
    result = AbstractArray(
        shape=keys.shape,
        dtype="float64",
    )
    return result

def witness_partialsortpartition(a: AbstractArray, kth: AbstractArray, axis: AbstractArray, kind: AbstractArray, order: AbstractArray) -> AbstractArray:
    """Ghost witness for PartialSortPartition."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result
