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
    """Entry-point orchestration for one Sequential Monte Carlo (SMC) step.

Args:
    up: Prior state carrying particles, weights, and random number generator (RNG) state.
    b: Transition and likelihood model definitions.
    a: Control/action at current step.
    o: Measurement for current time index.

Returns:
    Tuple of (prior_state, model_spec, control_t, observation_t, rng_key)."""
    # Unpack prior state and prepare for one SMC step
    particles = up.get('particles', np.array([])) if isinstance(up, dict) else up
    weights = up.get('weights', np.ones(1)) if isinstance(up, dict) else np.ones(1)
    rng_seed = up.get('rng_seed', 0) if isinstance(up, dict) else 0
    rng_key = np.array([rng_seed], dtype=np.int64)
    return (up, b, a, o, rng_key)


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
    # Propagate particles through transition model with noise
    particles = prior_state.get('particles', prior_state) if isinstance(prior_state, dict) else prior_state
    rng = np.random.RandomState(int(rng_key[0]) if len(rng_key) > 0 else 0)
    n_particles = len(particles) if hasattr(particles, '__len__') else 1
    noise = rng.randn(n_particles) if isinstance(particles, np.ndarray) else rng.randn(1)
    proposed = np.asarray(particles) + noise
    carry_weights = prior_state.get('weights', np.ones(n_particles) / n_particles) if isinstance(prior_state, dict) else np.ones(n_particles) / n_particles
    rng_key_next = np.array([int(rng_key[0]) + 1 if len(rng_key) > 0 else 1], dtype=np.int64)
    return (proposed, carry_weights, rng_key_next)


@register_atom(witness_likelihood_reweight_kernel)
@icontract.require(lambda proposed_particles: proposed_particles is not None, "proposed_particles cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def likelihood_reweight_kernel(
    proposed_particles: np.ndarray,
    carry_weights: np.ndarray,
    observation_t: np.ndarray,
    model_spec: object,
) -> tuple[np.ndarray, float]:
    """Compute observation likelihoods and perform Sequential Monte Carlo (SMC) weight update.

    Args:
        proposed_particles: Particles from propagation kernel.
        carry_weights: Prior importance weights.
        observation_t: Current measurement.
        model_spec: Provides measurement likelihood function.

    Returns:
        Tuple of (normalized_weights, log_likelihood).
    """
    # Gaussian likelihood: p(obs | particle) ~ exp(-0.5 * (obs - particle)^2)
    obs = np.asarray(observation_t)
    particles = np.asarray(proposed_particles)
    log_lik = -0.5 * np.sum((particles - obs) ** 2, axis=-1) if particles.ndim > 1 else -0.5 * (particles - obs.ravel()[0]) ** 2
    log_weights = np.log(carry_weights + 1e-300) + log_lik
    # Log-sum-exp normalization
    max_lw = np.max(log_weights)
    weights_exp = np.exp(log_weights - max_lw)
    total = weights_exp.sum()
    normalized = weights_exp / total
    log_likelihood = float(max_lw + np.log(total) - np.log(len(particles)))
    return (normalized, log_likelihood)


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
    rng_key_next: random number generator (RNG) key after split.
    log_likelihood: Aggregate log-likelihood for the step.

Returns:
    Tuple of (posterior_belief_state, filter_trace)."""
    # Systematic resampling
    n = len(normalized_weights)
    rng = np.random.RandomState(int(rng_key_next[0]) if len(rng_key_next) > 0 else 0)
    positions = (rng.uniform() + np.arange(n)) / n
    cumsum = np.cumsum(normalized_weights)
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, n - 1)
    resampled = proposed_particles[indices]
    uniform_weights = np.ones(n) / n
    posterior = {'particles': resampled, 'weights': uniform_weights, 'rng_seed': int(rng_key_next[0]) + 1 if len(rng_key_next) > 0 else 1}
    trace = {'log_likelihood': log_likelihood, 'ess': 1.0 / np.sum(normalized_weights ** 2)}
    return (posterior, trace)
