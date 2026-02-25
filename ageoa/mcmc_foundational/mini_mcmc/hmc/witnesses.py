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

def witness_initializesamplerandrngstate(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: InitializeSamplerAndRNGState."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_integratehamiltonianproposal(state_in: AbstractArray, log_prob_oracle: AbstractArray) -> AbstractArray:
    """Ghost witness for IntegrateHamiltonianProposal."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_applyhmcacceptrejectkernel(state_in: AbstractArray, proposal_state: AbstractArray, rng_state_in: AbstractArray) -> AbstractArray:
    """Ghost witness for ApplyHMCAcceptRejectKernel."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_collectposteriorsamples(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: CollectPosteriorSamples."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)
