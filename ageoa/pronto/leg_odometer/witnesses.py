from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_velocitystatereadout(state_in: AbstractArray) -> AbstractArray:
    """Ghost witness for VelocityStateReadout."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",)
    
    return result

def witness_posequeryaccessors() -> AbstractArray:
    """Ghost witness for PoseQueryAccessors."""
    return AbstractArray(shape=("6",), dtype="float64")