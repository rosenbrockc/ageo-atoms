"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations


try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_simulate_copula_dependence(returns: AbstractArray, rho: AbstractScalar, df: AbstractScalar) -> AbstractArray:
    """Ghost witness for simulate_copula_dependence."""
    result = AbstractArray(
        shape=returns.shape,
        dtype="float64",
    )
    return result
