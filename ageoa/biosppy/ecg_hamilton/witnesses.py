from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_hamilton_segmentation(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for hamilton_segmentation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
