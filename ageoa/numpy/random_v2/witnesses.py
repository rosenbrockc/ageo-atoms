"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
    from ageoa.ghost.abstract import AbstractMCMCTrace
    from ageoa.ghost.abstract import AbstractRNGState
except ImportError:
    pass

def witness_continuousmultivariatesampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: ContinuousMultivariateSampler."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_discreteeventsampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: DiscreteEventSampler."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_combinatoricssampler(x: AbstractArray, axis: AbstractArray, a: AbstractArray, size: AbstractArray, replace: AbstractArray, p: AbstractArray) -> AbstractArray:
    """Ghost witness for CombinatoricsSampler."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result
