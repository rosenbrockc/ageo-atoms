from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import witness_threshold_based_onset_detection
from biosppy.signals.emg import solnik_onset_detector

@register_atom(witness_threshold_based_onset_detection)
@icontract.require(lambda signal: signal.ndim >= 1, "signal must be at least 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be np.ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda threshold: isinstance(threshold, (float, int, np.number)), "threshold must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def threshold_based_onset_detection(signal: np.ndarray, rest: np.ndarray, sampling_rate: float, threshold: float, active_state_duration: float) -> np.ndarray:
    """Detect activation onset points in a signal by comparing against a rest-derived baseline and enforcing a minimum active-state duration.

    Args:
        signal: 1-D time-series samples.
        rest: Baseline/rest segment used to calibrate detection.
        sampling_rate: Sampling frequency in Hz; must be > 0.
        threshold: Activation cutoff relative to baseline/rest statistics.
        active_state_duration: Minimum sustained active duration in seconds.

    Returns:
        Indices marking detected onsets that satisfy threshold and duration criteria.
    """
    return solnik_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, threshold=threshold, active_state_duration=active_state_duration)["onsets"]
