from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_asi_signal_segmenter
from biosppy.signals.ecg import ASI_segmenter

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_asi_signal_segmenter)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda Pth: isinstance(Pth, (float, int, np.number)), "Pth must be numeric")
@icontract.ensure(lambda result: result is not None, "ASI_signal_segmenter output must not be None")
def asi_signal_segmenter(signal: np.ndarray, sampling_rate: float, Pth: float) -> np.ndarray:  # type: ignore[type-arg]
    """Segments an input signal into discrete intervals by applying a power/amplitude threshold (Pth) relative to the signal_primes sampling rate. Identifies contiguous regions where signal energy exceeds or falls below the threshold, returning segment boundary indices or masked signal regions.

    Args:
        signal: must be finite, length >= 1
        sampling_rate: sampling_rate > 0
        Pth: Pth > 0; determines the segmentation decision boundary

    Returns:
        0 <= start_sample < end_sample <= len(signal); non-overlapping, sorted ascending
    """
    return ASI_segmenter(signal=signal, sampling_rate=sampling_rate, Pth=Pth)["rpeaks"]
