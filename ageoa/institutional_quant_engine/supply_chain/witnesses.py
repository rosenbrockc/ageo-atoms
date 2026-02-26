"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations


try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_propagate_supply_shock(adjacency: AbstractArray, initial_shock: AbstractArray) -> AbstractArray:
    """Ghost witness for propagate_supply_shock."""
    result = AbstractArray(
        shape=adjacency.shape,
        dtype="float64",
    )
    return result
