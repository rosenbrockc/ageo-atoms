from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_fractional_differentiator(series: AbstractArray, d: AbstractScalar, threshold: AbstractScalar) -> AbstractArray:
    """Ghost witness for fractional_differentiator."""
    result = AbstractSignal(
        shape=series.shape,
        dtype="float64",
        sampling_rate=getattr(series, 'sampling_rate', 44100.0),
        domain="time",)
    
    return result