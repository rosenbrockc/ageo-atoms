"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_gamboa_segmentation)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: result is not None, "gamboa_segmentation output must not be None")
def gamboa_segmentation(signal: np.ndarray, sampling_rate: float, tol: float) -> np.ndarray:
    """Segments a signal into isoelectric and non-isoelectric regions based on the Gamboa (2008) method. This algorithm identifies segments by analyzing the standard deviation of the signal within a moving window.

    Args:
        signal: 1D array representing the signal.
        sampling_rate: The sampling frequency of the signal in Hz.
        tol: The tolerance for the segmentation, typically a small value like 0.001.

    Returns:
        An array of indices indicating the start and end points of the identified segments.
    """
    raise NotImplementedError("Wire to original implementation")
