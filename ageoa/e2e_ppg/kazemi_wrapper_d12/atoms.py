from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_normalizesignal, witness_wrapperevaluate


@register_atom(witness_normalizesignal)
@icontract.require(lambda arr: isinstance(arr, np.ndarray), "arr must be a numpy array")
@icontract.require(lambda arr: arr.size > 0, "arr must be non-empty")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "normalizesignal must return a numpy array")
def normalizesignal(arr: np.ndarray) -> np.ndarray:
    """Normalizes a raw array to a standard scale, producing a unit-normalized output suitable for downstream comparison or scoring.

    Args:
        arr: non-empty, finite values

    Returns:
        same shape as input, normalized between 0 and 1
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_wrapperevaluate)
@icontract.require(lambda prediction: isinstance(prediction, np.ndarray), "prediction must be a numpy array")
@icontract.require(lambda raw_signal: isinstance(raw_signal, np.ndarray), "raw_signal must be a numpy array")
@icontract.require(lambda normalized_arr: isinstance(normalized_arr, np.ndarray), "normalized_arr must be a numpy array")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "wrapperevaluate must return a numpy array")
def wrapperevaluate(prediction: np.ndarray, raw_signal: np.ndarray, normalized_arr: np.ndarray) -> np.ndarray:
    """Post-processes model predictions against the raw signal and normalized array to extract final peak indices.

    Args:
        prediction: model output predictions array
        raw_signal: non-empty, finite values; original raw signal
        normalized_arr: output of normalizesignal; values in [0, 1]

    Returns:
        Array of integer indices identifying detected peaks in the signal.
    """
    raise NotImplementedError("Wire to original implementation")
