"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_detect_onsets_with_rest_aware_thresholds(signal: AbstractSignal, rest: AbstractSignal, sampling_rate: AbstractSignal, size: AbstractSignal, alarm_size: AbstractSignal, threshold: AbstractSignal, transition_threshold: AbstractSignal) -> AbstractSignal:
    """Ghost witness for detect_onsets_with_rest_aware_thresholds."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
