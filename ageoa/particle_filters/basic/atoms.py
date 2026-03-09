from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import (
    witness_filter_step_preparation_and_dispatch,
    witness_particle_propagation_kernel,
    witness_likelihood_reweight_kernel,
    witness_resample_and_belief_projection,
)


@register_atom(witness_filter_step_preparation_and_dispatch)
@icontract.require(lambda up: up is not None, "prior state up cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def filter_step_preparation_and_dispatch(
    up: object, b: object, a: object, o: object
) -> tuple[object, object, object, object, np.ndarray]:
    """Entry-point orchestration for one SMC step.

    Args:
        up: Prior state carrying particles, weights, and RNG state.
        b: Transition and likelihood model definitions.
        a: Control/action at current step.
        o: Measurement for current time index.

    Returns:
        Tuple of (prior_state, model_spec, control_t, observation_t, rng_key).
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_particle_propagation_kernel)
@icontract.require(lambda prior_state: prior_state is not None, "prior_state cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def particle_propagation_kernel(
    prior_state: object, model_spec: object, control_t: object, rng_key: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate particles from prior to proposed latent state.

    Args:
        prior_state: Reads particles/weights only.
        model_spec: Provides transition dynamics.
        control_t: Current control/action.
        rng_key: Must be consumed and split functionally.

    Returns:
        Tuple of (proposed_particles, carry_weights, rng_key_next).
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_likelihood_reweight_kernel)
@icontract.require(lambda proposed_particles: proposed_particles is not None, "proposed_particles cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def likelihood_reweight_kernel(
    proposed_particles: np.ndarray,
    carry_weights: np.ndarray,
    observation_t: np.ndarray,
    model_spec: object,
) -> tuple[np.ndarray, float]:
    """Compute observation likelihoods and perform SMC weight update.

    Args:
        proposed_particles: Particles from propagation kernel.
        carry_weights: Prior importance weights.
        observation_t: Current measurement.
        model_spec: Provides measurement likelihood function.

    Returns:
        Tuple of (normalized_weights, log_likelihood).
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_resample_and_belief_projection)
@icontract.require(lambda log_likelihood: isinstance(log_likelihood, (float, int, np.number)), "log_likelihood must be numeric")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def resample_and_belief_projection(
    proposed_particles: np.ndarray,
    normalized_weights: np.ndarray,
    rng_key_next: np.ndarray,
    log_likelihood: float,
) -> tuple[object, object]:
    """Build posterior belief state and diagnostics from weighted particles.

    Args:
        proposed_particles: Particle array at current timestep.
        normalized_weights: Posterior normalized weights summing to 1.
        rng_key_next: RNG key after split.
        log_likelihood: Aggregate log-likelihood for the step.

    Returns:
        Tuple of (posterior_belief_state, filter_trace).
    """
    raise NotImplementedError("Wire to original implementation")
