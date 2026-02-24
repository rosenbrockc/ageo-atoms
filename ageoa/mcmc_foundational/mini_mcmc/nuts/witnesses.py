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

def witness_find_reasonable_epsilon(position: AbstractArray, mom: AbstractArray, gradient_target: AbstractArray) -> AbstractArray:
    """Ghost witness for Find Reasonable Epsilon."""
    result = AbstractArray(
        shape=position.shape,
        dtype="float64",
    )
    return result

def witness_build_tree(position: AbstractArray, mom: AbstractArray, grad: AbstractArray, logu: AbstractArray, v: AbstractArray, j: AbstractArray, epsilon: AbstractArray, gradient_target: AbstractArray, joint_0: AbstractArray, rng: AbstractArray) -> AbstractArray:
    """Ghost witness for Build Tree."""
    result = AbstractArray(
        shape=position.shape,
        dtype="float64",
    )
    return result

def witness_all_real(x: AbstractArray) -> AbstractArray:
    """Ghost witness for All Real."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_stop_criterion(position_minus: AbstractArray, position_plus: AbstractArray, mom_minus: AbstractArray, mom_plus: AbstractArray) -> AbstractArray:
    """Ghost witness for Stop Criterion."""
    result = AbstractArray(
        shape=position_minus.shape,
        dtype="float64",
    )
    return result

def witness_leapfrog(position: AbstractArray, mom: AbstractArray, grad: AbstractArray, epsilon: AbstractArray, gradient_target: AbstractArray) -> AbstractArray:
    """Ghost witness for Leapfrog."""
    result = AbstractArray(
        shape=position.shape,
        dtype="float64",
    )
    return result
