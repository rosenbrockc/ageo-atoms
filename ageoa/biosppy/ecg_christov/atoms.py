"""Auto-generated atom wrappers for christov_segmenter."""

from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from ageoa.biosppy.ecg_christov.witnesses import witness_christovqrsdetect


@register_atom(witness_christovqrsdetect)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be ndarray")
@icontract.require(lambda signal: signal.ndim == 1, "signal must be 1-D")
@icontract.require(lambda signal: np.isfinite(signal).all(), "signal must be finite")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int)), "sampling_rate must be numeric")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be ndarray")
@icontract.ensure(lambda result: result.ndim == 1, "result must be 1-D")
def christovqrsdetect(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Detect QRS complexes using the Christov real-time segmentation algorithm.

    Applies bandpass filtering, nonlinear energy operator, adaptive
    thresholding, and R-peak refinement to locate R-peak positions in an
    ECG signal.

    Args:
        signal: One-dimensional ECG signal array with finite real values.
        sampling_rate: Sampling rate in Hz, must be positive.

    Returns:
        Array of R-peak indices, monotonically increasing, values in
        [0, len(signal)-1].
    """
    raise NotImplementedError("Wire to original implementation")
