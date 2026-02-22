"""Ghost witnesses."""

from ageoa.ghost.abstract import AbstractArray

def witness_kazemi_peak_detection(data: AbstractArray) -> AbstractArray:
    """Witness for kazemi_peak_detection."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_ppg_reconstruction(data: AbstractArray) -> AbstractArray:
    """Witness for ppg_reconstruction."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_ppg_sqa(data: AbstractArray) -> AbstractArray:
    """Witness for ppg_sqa."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)
