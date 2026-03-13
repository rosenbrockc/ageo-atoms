from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_filterstateinit(b: AbstractArray, a: AbstractArray, state: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for FilterStateInit."""
    result = AbstractArray(
        shape=b.shape,
        dtype="float64",
    )
    return result, state

def witness_filterstep(signal: AbstractSignal, b: AbstractSignal, a: AbstractSignal, zi: AbstractSignal, state: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Ghost witness for FilterStep."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result, state
