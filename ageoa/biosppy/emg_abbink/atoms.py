from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import numpy.typing as npt

import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import witness_detect_onsets_with_rest_aware_thresholds

@register_atom(witness_detect_onsets_with_rest_aware_thresholds)
@icontract.require(lambda signal: signal.ndim >= 1, "signal must be at least 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be np.ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda threshold: isinstance(threshold, (float, int, np.number)), "threshold must be numeric")
@icontract.require(lambda transition_threshold: isinstance(transition_threshold, (float, int, np.number)), "transition_threshold must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def detect_onsets_with_rest_aware_thresholds(signal: npt.NDArray[np.float64], rest: npt.NDArray[np.float64] | float | int, sampling_rate: float, size: int, alarm_size: int, threshold: float, transition_threshold: float) -> npt.NDArray[np.float64]:
    """Detect onset events from an input signal using rest/reference information, window sizes, sampling rate, and transition/alarm thresholds.

    Args:
        signal: Input signal; non-empty, length should support window operations.
        rest: Reference signal or scalar compatible with signal domain.
        sampling_rate: Sampling frequency in Hz; must be > 0.
        size: Window size; must be > 0.
        alarm_size: Alarm window size; must be > 0.
        threshold: Application-defined detection threshold.
        transition_threshold: Application-defined transition threshold.

    Returns:
        Onset detection results aligned with input signal timeline.
    """
    raise NotImplementedError("Wire to original implementation")
