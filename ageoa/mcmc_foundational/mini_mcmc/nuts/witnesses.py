"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
except ImportError:
    pass

def witness_initialize_sampler(event_shape: tuple[int, ...], family: str = "normal") -> AbstractDistribution:
    """Ghost witness for prior init: initialize_sampler."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,
    )

def witness_run_mcmc_sampler(sampler_state_in: AbstractArray, n_collect: AbstractArray, n_discard: AbstractArray) -> AbstractArray:
    """Ghost witness for run_mcmc_sampler."""
    result = AbstractArray(
        shape=sampler_state_in.shape,
        dtype="float64",
    )
    return result
