from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_forwardmultidimensionalfft(a: AbstractSignal, s: AbstractSignal, axes: AbstractSignal, norm: AbstractSignal) -> AbstractSignal:
    """Ghost witness for ForwardMultidimensionalFFT."""
    result = AbstractSignal(
        shape=a.shape,
        dtype="float64",
        sampling_rate=getattr(a, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_inversemultidimensionalfft(a: AbstractSignal, s: AbstractSignal, axes: AbstractSignal, norm: AbstractSignal) -> AbstractSignal:
    """Ghost witness for InverseMultidimensionalFFT."""
    result = AbstractSignal(
        shape=a.shape,
        dtype="float64",
        sampling_rate=getattr(a, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_hermitianspectraltransform(a: AbstractSignal, n: AbstractSignal, axis: AbstractSignal, norm: AbstractSignal) -> AbstractSignal:
    """Ghost witness for HermitianSpectralTransform."""
    result = AbstractSignal(
        shape=a.shape,
        dtype="float64",
        sampling_rate=getattr(a, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
