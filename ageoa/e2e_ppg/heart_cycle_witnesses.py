from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_detect_heart_cycles(ppg: AbstractSignal, sampling_rate: AbstractScalar) -> AbstractSignal:
    """Shape-and-type check for detect heart cycles. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=ppg.shape,
        dtype="float64",
        sampling_rate=getattr(ppg, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_heart_cycle_detection(ppg: AbstractArray, sampling_rate: AbstractArray) -> AbstractArray:
    """Shape-and-type check for heart cycle detection. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=ppg.shape,
        dtype="float64",
    )
    return result
