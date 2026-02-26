"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
    from ageoa.ghost.abstract import AbstractMCMCTrace
    from ageoa.ghost.abstract import AbstractRNGState
except ImportError:
    pass

def witness_filter_step_preparation_and_dispatch(up: AbstractArray, b: AbstractArray, a: AbstractArray, o: AbstractArray) -> AbstractArray:
    """Ghost witness for Filter Step Preparation And Dispatch."""
    result = AbstractArray(
        shape=up.shape,
        dtype="float64",
    )
    return result

def witness_particle_propagation_kernel(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: Particle Propagation Kernel."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_likelihood_reweight_kernel(proposed_particles: AbstractArray, carry_weights: AbstractArray, observation_t: AbstractArray, model_spec: AbstractArray) -> AbstractArray:
    """Ghost witness for Likelihood Reweight Kernel."""
    result = AbstractArray(
        shape=proposed_particles.shape,
        dtype="float64",
    )
    return result

def witness_resample_and_belief_projection(prior: AbstractDistribution, likelihood: AbstractDistribution, data_shape: tuple[int, ...]) -> AbstractDistribution:
    """Ghost witness for posterior update: Resample And Belief Projection."""
    prior.assert_conjugate_to(likelihood)
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,
    )
