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

def witness_riskaversioninit(risk_aversion: AbstractArray) -> AbstractArray:
    """Ghost witness for RiskAversionInit."""
    result = AbstractArray(
        shape=risk_aversion.shape,
        dtype="float64",
    )
    return result

def witness_optimalexecutiontrajectory(risk_aversion: AbstractArray, total_shares: AbstractArray, days: AbstractArray) -> AbstractArray:
    """Ghost witness for OptimalExecutionTrajectory."""
    result = AbstractArray(
        shape=risk_aversion.shape,
        dtype="float64",
    )
    return result
