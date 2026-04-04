from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray

def witness_velocitystatereadout(state_in: AbstractArray) -> AbstractArray:
    """Describe the velocity-vector/covariance readout from immutable odometry state."""
    return AbstractArray(shape=("2",), dtype="tuple[ndarray,float64]")

def witness_posequeryaccessors() -> AbstractArray:
    """Describe the stateless position/orientation query surface."""
    return AbstractArray(shape=("6",), dtype="float64")
