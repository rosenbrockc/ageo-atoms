from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_normalizesignal(arr: AbstractSignal) -> AbstractSignal:
    """Ghost witness for NormalizeSignal."""
    result = AbstractSignal(
        shape=arr.shape,
        dtype="float64",
        sampling_rate=getattr(arr, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_wrapperevaluate(prediction: AbstractArray, raw_signal: AbstractArray, normalized_arr: AbstractArray) -> AbstractArray:
    """Ghost witness for WrapperEvaluate."""
    result = AbstractArray(
        shape=prediction.shape,
        dtype="float64",
    )
    return result
