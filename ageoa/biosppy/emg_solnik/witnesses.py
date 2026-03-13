from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_threshold_based_onset_detection(signal: AbstractSignal,
    rest: AbstractSignal,
    sampling_rate: AbstractScalar,
    threshold: AbstractScalar,
    active_state_duration: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for threshold-based onset detection. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
