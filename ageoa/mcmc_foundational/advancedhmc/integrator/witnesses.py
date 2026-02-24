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

def witness_temper(lf: AbstractArray, r: AbstractArray) -> AbstractArray:
    """Ghost witness for Temper."""
    result = AbstractArray(
        shape=lf.shape,
        dtype="float64",
    )
    return result

def witness_show(io: AbstractArray, l: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_show(io: AbstractArray, l: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_jitter(rng: AbstractArray, lf: AbstractArray) -> AbstractArray:
    """Ghost witness for Jitter."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_jitter(rng: AbstractArray, lf: AbstractArray) -> AbstractArray:
    """Ghost witness for Jitter."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_show(io: AbstractArray, l: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_temper(lf: AbstractArray, r: AbstractArray, step: AbstractArray, n_steps: AbstractArray) -> AbstractArray:
    """Ghost witness for Temper."""
    result = AbstractArray(
        shape=lf.shape,
        dtype="float64",
    )
    return result

def witness_step(lf: AbstractArray, h: AbstractArray, z: AbstractArray) -> AbstractArray:
    """Ghost witness for Step."""
    result = AbstractArray(
        shape=lf.shape,
        dtype="float64",
    )
    return result
