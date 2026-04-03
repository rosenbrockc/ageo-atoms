from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal, AbstractMCMCTrace, AbstractRNGState


def witness_filter_step_preparation_and_dispatch(up: AbstractArray, b: AbstractArray, a: AbstractArray, o: AbstractArray) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Shape-and-type check for filter step preparation and dispatch. Returns output metadata without running the real computation."""
    rng_key = AbstractArray(
        shape=(1,),
        dtype="int64",
    )
    return up, b, a, o, rng_key

def witness_particle_propagation_kernel(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Shape-and-type check for mcmc sampler: particle propagation kernel. Returns output metadata without running the real computation."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_likelihood_reweight_kernel(proposed_particles: AbstractArray, carry_weights: AbstractArray, observation_t: AbstractArray, model_spec: AbstractArray) -> AbstractArray:
    """Shape-and-type check for likelihood reweight kernel. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=proposed_particles.shape,
        dtype="float64",
    )
    return result

def witness_resample_and_belief_projection(prior: AbstractDistribution, likelihood: AbstractDistribution, data_shape: tuple[int, ...]) -> AbstractDistribution:
    """Shape-and-type check for posterior update: resample and belief projection. Returns output metadata without running the real computation."""
    prior.assert_conjugate_to(likelihood)
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,
    )
