from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal, ANYTHING

def witness_sample_hawkes_event_trajectory(
    target: AbstractDistribution,
    trace: AbstractMCMCTrace,
    rng: AbstractRNGState,
    *args, **kwargs
) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Shape-and-type check for mcmc sampler: sample hawkes event trajectory. Returns output metadata without running the real computation."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)
