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

def witness_foot_sensing_state_update(foot_sensing_state_in: AbstractArray, foot_sensing_command: AbstractArray) -> AbstractArray:
    """Ghost witness for Foot Sensing State Update."""
    result = AbstractArray(
        shape=foot_sensing_state_in.shape,
        dtype="float64",
    )
    return result

def witness_mode_snapshot_readout(mode_state_in: AbstractArray) -> AbstractArray:
    """Ghost witness for Mode Snapshot Readout."""
    result = AbstractArray(
        shape=mode_state_in.shape,
        dtype="float64",
    )
    return result
