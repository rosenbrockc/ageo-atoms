from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_wrapperpredictionsignalcomputation(prediction: AbstractArray, raw_signal: AbstractArray) -> AbstractArray:
    """Ghost witness for WrapperPredictionSignalComputation."""
    result = AbstractArray(
        shape=prediction.shape,
        dtype="float64",
    )
    return result

def witness_signalarraynormalization(arr: AbstractArray) -> AbstractArray:
    """Ghost witness for SignalArrayNormalization."""
    result = AbstractArray(
        shape=arr.shape,
        dtype="float64",
    )
    return result
