from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_kazemi_peak_detection(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for kazemi peak detection. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_ppg_reconstruction(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for ppg reconstruction. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_ppg_sqa(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for ppg sqa. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)
