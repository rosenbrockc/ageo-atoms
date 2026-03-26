from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_detect_onsets_with_rest_aware_thresholds(signal: AbstractSignal,
    rest: AbstractSignal,
    sampling_rate: AbstractScalar,
    size: AbstractScalar,
    alarm_size: AbstractScalar,
    threshold: AbstractScalar,
    transition_threshold: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for detect onsets with rest aware thresholds. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
