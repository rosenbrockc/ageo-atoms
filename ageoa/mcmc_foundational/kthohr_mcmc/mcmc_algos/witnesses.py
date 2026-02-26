"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations


try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_dispatch_mcmc_algorithm(log_target_density: AbstractArray, initial_state: AbstractArray, n_draws: AbstractScalar) -> AbstractArray:
    """Ghost witness for dispatch_mcmc_algorithm."""
    result = AbstractArray(
        shape=log_target_density.shape,
        dtype="float64",
    )
    return result
