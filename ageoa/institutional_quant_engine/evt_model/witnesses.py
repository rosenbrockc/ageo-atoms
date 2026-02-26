"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations


try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_fit_gpd_tail(returns: AbstractArray, threshold_quantile: AbstractScalar) -> AbstractArray:
    """Ghost witness for fit_gpd_tail."""
    result = AbstractArray(
        shape=returns.shape,
        dtype="float64",
    )
    return result
