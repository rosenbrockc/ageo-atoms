from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_initializebacklashfilterstate() -> AbstractArray:
    """Shape-and-type check for initialize backlash filter state. Returns output metadata without running the real computation."""
    return AbstractArray(shape=("S",), dtype="float64")

def witness_updatealphaparameter(state_in: AbstractArray, alpha_in: AbstractArray) -> AbstractArray:
    """Shape-and-type check for update alpha parameter. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",)
    
    return result

def witness_updatecrossingtimemaximum(state_in: AbstractArray, t_crossing_max_in: AbstractArray) -> AbstractArray:
    """Shape-and-type check for update crossing time maximum. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",)
    
    return result