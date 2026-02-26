"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_initializebacklashfilterstate() -> AbstractArray:
    """Ghost witness for InitializeBacklashFilterState."""
    return AbstractArray(shape=("S",), dtype="float64")

def witness_updatealphaparameter(state_in: AbstractArray, alpha_in: AbstractArray) -> AbstractArray:
    """Ghost witness for UpdateAlphaParameter."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_updatecrossingtimemaximum(state_in: AbstractArray, t_crossing_max_in: AbstractArray) -> AbstractArray:
    """Ghost witness for UpdateCrossingTimeMaximum."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result
