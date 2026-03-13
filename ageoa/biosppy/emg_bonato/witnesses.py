from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_bonato_onset_detection(signal: AbstractSignal,
    rest: AbstractSignal,
    sampling_rate: AbstractScalar,
    threshold: AbstractScalar,
    active_state_duration: AbstractScalar,
    samples_above_fail: AbstractScalar,
    fail_size: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for bonato onset detection. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
