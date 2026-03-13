from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal, ANYTHING

def witness_sample_hawkes_event_trajectory(
    target: AbstractDistribution,
    trace: AbstractMCMCTrace,
    rng: AbstractRNGState,
    *args, **kwargs
) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: sample_hawkes_event_trajectory."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)
