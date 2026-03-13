from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_gamboa_segmentation(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    tol: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for gamboa_segmentation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
