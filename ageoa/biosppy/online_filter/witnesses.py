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

def witness_filterstateinit(b: AbstractArray, a: AbstractArray, state: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for FilterStateInit."""
    result = AbstractArray(
        shape=b.shape,
        dtype="float64",
    )
    return result, state

def witness_filterstep(signal: AbstractSignal, b: AbstractSignal, a: AbstractSignal, zi: AbstractSignal, state: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Ghost witness for FilterStep."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result, state
