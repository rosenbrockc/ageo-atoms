"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
    from ageoa.ghost.abstract import AbstractMCMCTrace
    from ageoa.ghost.abstract import AbstractRNGState
except ImportError:
    pass

def witness_initializenutsstate(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: InitializeNUTSState."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_runnutstransitions(nuts_state_in: AbstractArray, rng_key_in: AbstractArray, n_collect: AbstractArray, n_discard: AbstractArray) -> AbstractArray:
    """Ghost witness for RunNUTSTransitions."""
    result = AbstractArray(
        shape=nuts_state_in.shape,
        dtype="float64",
    )
    return result
