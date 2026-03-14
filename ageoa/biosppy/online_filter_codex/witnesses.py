from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_filterstateinit(b: AbstractArray, a: AbstractArray, state: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for filter state init. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=b.shape,
        dtype="float64",
    )
    return result, state

def witness_filterstep(signal: AbstractSignal, b: AbstractSignal, a: AbstractSignal, zi: AbstractSignal, state: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Shape-and-type check for filter step. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result, state
