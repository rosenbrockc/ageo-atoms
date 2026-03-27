from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from biosppy.signals.ecg import ASI_segmenter
# from .ecg_detectors_witnesses import *

# Witness functions should be imported from the generated witnesses module

@register_atom(lambda *args, **kwargs: None)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda Pth: isinstance(Pth, (float, int, np.number)), "Pth must be numeric")
@icontract.ensure(lambda result: result is not None, "ThresholdBasedSignalSegmentation output must not be None")
def thresholdbasedsignalsegmentation(signal: np.ndarray, sampling_rate: float, Pth: float) -> np.ndarray:
    """Segments the input signal into activity regions using the provided sampling rate and decision threshold.

    Args:
        signal: 1-D or compatible signal shape; finite values preferred
        sampling_rate: must be > 0
        Pth: threshold parameter used for segmentation decision

    Returns:
        derived deterministically from inputs
    """
    return ASI_segmenter(signal=signal, sampling_rate=sampling_rate, Pth=Pth)["rpeaks"]

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .ecg_detectors_witnesses import witness_asi_signal_segmenter
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

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .ecg_detectors_witnesses import witness_christovqrsdetect
from biosppy.signals.ecg import christov_segmenter


@register_atom(witness_christovqrsdetect)
@icontract.require(lambda signal: np.isfinite(signal).all(), "signal must be finite")
@icontract.require(lambda signal: signal.ndim == 1, "signal must be 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int)), "sampling_rate must be numeric")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be ndarray")
@icontract.ensure(lambda result: result.ndim == 1, "result must be 1-D")
def christovqrsdetect(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Detect QRS complex (the sharp spike marking each heartbeat) complexes using the Christov real-time segmentation algorithm.

Applies bandpass filtering, nonlinear energy operator, adaptive
thresholding, and R-peak refinement to locate R-peak positions in an
electrocardiogram (ECG) signal.

Args:
    signal: One-dimensional ECG signal array with finite real values.
    sampling_rate: Sampling rate in Hz, must be positive.

Returns:
    Array of R-peak indices, monotonically increasing, values in
    [0, len(signal)-1]."""
    return christov_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"]

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from biosppy.signals.ecg import christov_segmenter
# from .ecg_detectors_witnesses import witness_christov_qrs_segmenter

# Witness functions should be imported from the generated witnesses module

@register_atom("witness_christov_qrs_segmenter")  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "christov_qrs_segmenter output must not be None")
def christov_qrs_segmenter(signal: np.ndarray, sampling_rate: float) -> np.ndarray:  # type: ignore[type-arg]
    """Detects QRS complex (the sharp spike marking each heartbeat) complex onset and offset positions in an electrocardiogram (ECG) signal using the Christov real-time algorithm, which applies a series of signal transformations and adaptive thresholding to locate heartbeat boundaries.

Args:
    signal: must be a non-empty array of finite real values
    sampling_rate: must be > 0

Returns:
    indices must be within bounds of the input signal; sorted ascending"""
    return christov_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"]

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .ecg_detectors_witnesses import witness_engzee_signal_segmentation
from biosppy.signals.ecg import engzee_segmenter


@register_atom(witness_engzee_signal_segmentation)
@icontract.require(lambda signal: np.isfinite(signal).all(), "signal must be finite")
@icontract.require(lambda signal: signal.ndim == 1, "signal must be 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int)), "sampling_rate must be numeric")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.require(lambda threshold: isinstance(threshold, (float, int)), "threshold must be numeric")
@icontract.require(lambda threshold: 0.0 < threshold < 1.0, "threshold must be in (0, 1)")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be ndarray")
@icontract.ensure(lambda result: result.ndim == 1, "result must be 1-D")
def engzee_signal_segmentation(signal: np.ndarray, sampling_rate: float, threshold: float) -> np.ndarray:
    """Detect QRS complex (the sharp spike marking each heartbeat) complexes using the Engelse-Zeelenberg algorithm.

Applies threshold-intersection peak detection with consecutive-sample
validation to locate R-peak positions in an electrocardiogram (ECG) signal.

Args:
    signal: One-dimensional ECG signal array with finite real values.
    sampling_rate: Sampling rate in Hz, must be positive.
    threshold: Decision threshold in (0, 1) for peak identification.

Returns:
    Array of R-peak indices, monotonically increasing, values in
    [0, len(signal)-1]."""
    return engzee_segmenter(signal=signal, sampling_rate=sampling_rate, threshold=threshold)["rpeaks"]

from typing import Any

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .ecg_detectors_witnesses import witness_engzee_qrs_segmentation  # type: ignore[import-untyped]
from biosppy.signals.ecg import engzee_segmenter

@register_atom(witness_engzee_qrs_segmentation)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "engzee_qrs_segmentation output must not be None")
def engzee_qrs_segmentation(signal: np.ndarray, sampling_rate: float, threshold: float) -> np.ndarray:
    """Detects and segments QRS complex (the sharp spike marking each heartbeat) complexes from a raw electrocardiogram (ECG) signal using the Engelse & Zeelenberg algorithm, applying a threshold-based decision rule on the transformed signal to locate R-peak positions and extract beat boundaries.

    Args:
        signal: must be a uniformly sampled real-valued sequence
        sampling_rate: must be > 0
        threshold: detection sensitivity threshold; must be > 0

    Returns:
        indices into the input signal where R-peaks are detected; sorted ascending
    """
    return engzee_segmenter(signal=signal, sampling_rate=sampling_rate, threshold=threshold)["rpeaks"]

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .ecg_detectors_witnesses import witness_gamboa_segmentation
from biosppy.signals.ecg import gamboa_segmenter

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
    return gamboa_segmenter(signal=signal, sampling_rate=sampling_rate, tol=tol)["rpeaks"]

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .ecg_detectors_witnesses import witness_gamboa_segmenter
from biosppy.signals.ecg import gamboa_segmenter as _gamboa_segmenter

  # type: ignore[import-untyped]
@register_atom(witness_gamboa_segmenter)
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: result is not None, "gamboa_segmenter output must not be None")
def gamboa_segmenter(signal: np.ndarray, sampling_rate: float, tol: float) -> np.ndarray:
    """
    Args:
        signal: non-empty, finite values
        sampling_rate: must be > 0
        tol: must be > 0

    Returns:
        indices within [0, len(signal))
    """
    return _gamboa_segmenter(signal=signal, sampling_rate=sampling_rate, tol=tol)["rpeaks"]

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .ecg_detectors_witnesses import witness_hamilton_segmentation
from biosppy.signals.ecg import hamilton_segmenter

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_hamilton_segmentation)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.ensure(lambda result: result is not None, "hamilton_segmentation output must not be None")
def hamilton_segmentation(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Performs electrocardiogram (ECG) signal segmentation to detect QRS complex (the sharp spike marking each heartbeat) complexes using the Hamilton algorithm.

Args:
    signal: 1D array representing the ECG signal.
    sampling_rate: The sampling rate of the signal in Hz.

Returns:
    Array of indices corresponding to the detected R-peaks."""
    return hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"]

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from biosppy.signals.ecg import hamilton_segmenter as _hamilton_segmenter
# from .ecg_detectors_witnesses import witness_hamilton_segmenter

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
