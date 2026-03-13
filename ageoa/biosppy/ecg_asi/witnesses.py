from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_thresholdbasedsignalsegmentation(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    Pth: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for ThresholdBasedSignalSegmentation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
