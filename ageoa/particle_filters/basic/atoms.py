"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from typing import Any, Callable, TypeAlias, TypeVar, cast

from ageoa.ghost.registry import register_atom as _register_atom  # type: ignore[import-untyped]
from juliacall import Main as jl  # type: ignore[import-untyped]

F = TypeVar("F", bound=Callable[..., Any])


def register_atom(witness: object) -> Callable[[F], F]:
    return cast(Callable[[F], F], _register_atom(witness))


WeightedParticleBeliefState: TypeAlias = Any
StateModelSpec: TypeAlias = Any
ControlInput: TypeAlias = Any
Observation: TypeAlias = Any
PRNGKey: TypeAlias = Any
FilterTrace: TypeAlias = Any
Array: TypeAlias = Any
N: TypeAlias = Any
D_latent: TypeAlias = Any

witness_filter_step_preparation_and_dispatch: object = object()
witness_particle_propagation_kernel: object = object()
witness_likelihood_reweight_kernel: object = object()
witness_resample_and_belief_projection: object = object()
@register_atom(witness_filter_step_preparation_and_dispatch)
@icontract.require(lambda up: up is not None, "up cannot be None")
@icontract.require(lambda b: b is not None, "b cannot be None")
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda o: o is not None, "o cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "Filter Step Preparation And Dispatch all outputs must not be None")
def filter_step_preparation_and_dispatch(up: WeightedParticleBeliefState, b: StateModelSpec, a: ControlInput, o: Observation) -> tuple[WeightedParticleBeliefState, StateModelSpec, ControlInput, Observation, PRNGKey]:
    """Entry-point orchestration for one SMC step: normalize raw inputs and expose immutable state-space components for downstream pure kernels.

    Args:
        up: Immutable prior state carrying particles, weights, optional ancestors, and PRNGKey/RNG state.
        b: Transition and likelihood model definitions used by predict/reweight.
        a: Control/action at current step; may be empty for uncontrolled dynamics.
        o: Measurement for current time index.

    Returns:
        prior_state: Unmodified prior state object.
        model_spec: Pure model bundle forwarded unchanged.
        control_t: Preprocessed control/action.
        observation_t: Preprocessed observation.
        rng_key: Explicit stochastic key threaded to kernels; never implicit global mutation.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_particle_propagation_kernel)
@icontract.require(lambda prior_state: prior_state is not None, "prior_state cannot be None")
@icontract.require(lambda model_spec: model_spec is not None, "model_spec cannot be None")
@icontract.require(lambda control_t: control_t is not None, "control_t cannot be None")
@icontract.require(lambda rng_key: rng_key is not None, "rng_key cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "Particle Propagation Kernel all outputs must not be None")
def particle_propagation_kernel(prior_state: WeightedParticleBeliefState, model_spec: StateModelSpec, control_t: ControlInput, rng_key: PRNGKey) -> tuple[Array, Array, PRNGKey]:
    """Pure transition/proposal kernel that propagates particles from prior to proposed latent state and returns a split RNG key.

    Args:
        prior_state: Reads particles/weights only; does not mutate input.
        model_spec: Provides transition dynamics.
        control_t: Current control/action.
        rng_key: Must be consumed and split functionally.

    Returns:
        proposed_particles: New latent particle set at time t.
        carry_weights: Prior weights carried forward for reweighting.
        rng_key_next: New key object after split.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_likelihood_reweight_kernel)
@icontract.require(lambda proposed_particles: proposed_particles is not None, "proposed_particles cannot be None")
@icontract.require(lambda carry_weights: carry_weights is not None, "carry_weights cannot be None")
@icontract.require(lambda observation_t: observation_t is not None, "observation_t cannot be None")
@icontract.require(lambda model_spec: model_spec is not None, "model_spec cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "Likelihood Reweight Kernel all outputs must not be None")
def likelihood_reweight_kernel(proposed_particles: Array, carry_weights: Array, observation_t: Observation, model_spec: StateModelSpec) -> tuple[Array, float]:
    """Computes observation likelihoods/log-probs for proposed particles and performs immutable SMC weight update/normalization.

    Args:
        proposed_particles: Particles output by propagation kernel.
        carry_weights: Prior importance weights.
        observation_t: Current measurement.
        model_spec: Provides measurement likelihood function.

    Returns:
        normalized_weights: Posterior normalized weights (sum to 1).
        log_likelihood: Aggregate log-likelihood / evidence contribution for the step.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_resample_and_belief_projection)
@icontract.require(lambda log_likelihood: isinstance(log_likelihood, (float, int, np.number)), "log_likelihood must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "Resample And Belief Projection all outputs must not be None")
def resample_and_belief_projection(proposed_particles: Array, normalized_weights: Array, rng_key_next: PRNGKey, log_likelihood: float) -> tuple[WeightedParticleBeliefState, FilterTrace]:
    """Build posterior belief state and diagnostics from weighted particles."""
    raise NotImplementedError("Wire to original implementation")


def filter_step_preparation_and_dispatch_ffi(up: Any, b: Any, a: Any, o: Any) -> Any:
    """FFI bridge to Julia implementation of Filter Step Preparation And Dispatch."""
    return jl.eval("filter_step_preparation_and_dispatch(up, b, a, o)")


def particle_propagation_kernel_ffi(prior_state: Any, model_spec: Any, control_t: Any, rng_key: Any) -> Any:
    """FFI bridge to Julia implementation of Particle Propagation Kernel."""
    return jl.eval("particle_propagation_kernel(prior_state, model_spec, control_t, rng_key)")


def likelihood_reweight_kernel_ffi(proposed_particles: Any, carry_weights: Any, observation_t: Any, model_spec: Any) -> Any:
    """FFI bridge to Julia implementation of Likelihood Reweight Kernel."""
    return jl.eval("likelihood_reweight_kernel(proposed_particles, carry_weights, observation_t, model_spec)")


def resample_and_belief_projection_ffi(proposed_particles: Any, normalized_weights: Any, rng_key_next: Any, log_likelihood: Any) -> Any:
    """FFI bridge to Julia implementation of Resample And Belief Projection."""
    return jl.eval("resample_and_belief_projection(proposed_particles, normalized_weights, rng_key_next, log_likelihood)")