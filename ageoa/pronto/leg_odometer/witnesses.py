from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_velocitystatereadout(state_in: AbstractArray) -> AbstractArray:
    """Shape-and-type check for velocity state readout. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",)
    
    return result

def witness_posequeryaccessors() -> AbstractArray:
    """Shape-and-type check for pose query accessors. Returns output metadata without running the real computation."""
    return AbstractArray(shape=("6",), dtype="float64")