from __future__ import annotations

import numpy as np
import numpy.typing as npt

import icontract
from ageoa.ghost.registry import register_atom

from .emg_detectors_witnesses import witness_detect_onsets_with_rest_aware_thresholds
from biosppy.signals.emg import abbink_onset_detector

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
    return abbink_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, size=size, alarm_size=alarm_size, threshold=threshold, transition_threshold=transition_threshold)["onsets"]

from typing import List, Any, Optional, Tuple, Dict
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .emg_detectors_witnesses import witness_bonato_onset_detection
from biosppy.signals.emg import bonato_onset_detector

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_bonato_onset_detection)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.require(lambda rest: isinstance(rest, np.ndarray), "rest must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda threshold: isinstance(threshold, (float, int, np.number)), "threshold must be numeric")
@icontract.require(lambda active_state_duration: isinstance(active_state_duration, (float, int, np.number)), "active_state_duration must be numeric")
@icontract.ensure(lambda result: result is not None, "bonato_onset_detection output must not be None")
def bonato_onset_detection(signal: np.ndarray, rest: np.ndarray, sampling_rate: float, threshold: float, active_state_duration: float, samples_above_fail: int, fail_size: int) -> List[int]:
    """Detects activity onsets in a signal using the Bonato double-threshold algorithm. It identifies points where the signal exceeds a defined threshold for a minimum duration.

    Args:
        signal: Input signal trace as a 1D numpy array.
        rest: Signal segment corresponding to a rest period, used for noise estimation.
        sampling_rate: The signal_primes sampling frequency in Hz.
        threshold: The amplitude threshold for marking a potential onset.
        active_state_duration: The minimum duration (in seconds) an active state must be maintained to be confirmed.
        samples_above_fail: Number of consecutive samples that must be above the threshold to initiate an active state.
        fail_size: Size of the window (in samples) to check for validation after a potential onset.

    Returns:
        A list of integer indices corresponding to the detected onset locations in the signal.
    """
    return bonato_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, threshold=threshold, active_state_duration=active_state_duration, samples_above_fail=samples_above_fail, fail_size=fail_size)["onsets"]

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .emg_detectors_witnesses import witness_threshold_based_onset_detection
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

import icontract
"""Auto-generated atom wrappers following the ageoa pattern."""


from typing import Any
import numpy as np

from ageoa.ghost.registry import register_atom
from .emg_detectors_witnesses import witness_solnik_onset_detect
from biosppy.signals.emg import solnik_onset_detector
@register_atom(witness_solnik_onset_detect)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda rest: rest is not None, "rest cannot be None")
@icontract.require(lambda sampling_rate: sampling_rate is not None, "sampling_rate cannot be None")
@icontract.require(lambda threshold: threshold is not None, "threshold cannot be None")
@icontract.require(lambda active_state_duration: active_state_duration is not None, "active_state_duration cannot be None")
@icontract.ensure(lambda result: result is not None, "solnik_onset_detect output must not be None")
def solnik_onset_detect(signal: np.ndarray, rest: float, sampling_rate: float, threshold: float, active_state_duration: float) -> np.ndarray:
    """Detects movement onsets in a signal using the Solnik algorithm: identifies transitions from rest to active state by comparing signal amplitude against a threshold over a minimum active-state duration window.

    Args:
        signal: non-empty, finite values
        rest: finite, typically >= 0
        sampling_rate: > 0
        threshold: > 0
        active_state_duration: > 0

    Returns:
        values in [0, len(signal)-1], monotonically increasing
    """
    return solnik_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, threshold=threshold, active_state_duration=active_state_duration)["onsets"]
