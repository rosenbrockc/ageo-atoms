from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""



import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from biosppy.signals.ecg import hamilton_segmenter as _hamilton_segmenter
# from .witnesses import witness_hamilton_segmenter

# Witness functions should be imported from the generated witnesses module

@register_atom("witness_hamilton_segmenter")  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "hamilton_segmenter output must not be None")
def hamilton_segmenter(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Detects R-peaks (QRS complexes) in an electrocardiogram (ECG) signal using the Hamilton segmentation algorithm, returning the indices of detected peaks given a raw signal and its sampling rate.

    Args:
        signal: non-empty, numeric
        sampling_rate: must be > 0

    Returns:
        indices within [0, len(signal))
    """
    return _hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"]
