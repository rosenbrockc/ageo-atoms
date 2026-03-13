from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_homomorphic_signal_filtering(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for homomorphic_signal_filtering."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
