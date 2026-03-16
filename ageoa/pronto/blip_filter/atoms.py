from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_bandpass_filter, witness_heart_rate_computation, witness_peak_correction, witness_r_peak_detection, witness_template_extraction

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_bandpass_filter)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.ensure(lambda result: result is not None, "Bandpass Filter output must not be None")
def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """Apply a bandpass filter (3-45 Hz) to remove slow drift and high-frequency noise from the raw electrocardiogram (ECG) signal.

    Args:
        signal: 1D raw ECG signal

    Returns:
        bandpass-filtered ECG
    """
    from scipy.signal import butter, sosfiltfilt
    # 3-45 Hz bandpass for ECG, assume 360 Hz sampling rate as default
    fs = 360.0
    low, high = 3.0, 45.0
    sos = butter(4, [low / (fs / 2.0), high / (fs / 2.0)], btype='band', output='sos')
    return sosfiltfilt(sos, signal).astype(signal.dtype)

@register_atom(witness_r_peak_detection)
@icontract.require(lambda filtered: isinstance(filtered, np.ndarray), "filtered must be a numpy array")
@icontract.ensure(lambda result: result is not None, "R-Peak Detection output must not be None")
def r_peak_detection(filtered: np.ndarray) -> np.ndarray:
    """Detect R-peak locations — the prominent upward spikes in each heartbeat — in the filtered electrocardiogram (ECG) signal using the Hamilton segmenter (a threshold-based peak-finding algorithm).

    Args:
        filtered: filtered ECG signal

    Returns:
        R-peak sample indices
    """
    from scipy.signal import find_peaks
    # Hamilton-style R-peak detection: threshold at 60% of max amplitude
    abs_sig = np.abs(filtered)
    threshold = 0.6 * np.max(abs_sig)
    # Minimum distance between peaks: ~200ms at 360 Hz = 72 samples
    min_dist = max(1, int(0.2 * 360))
    peaks, _ = find_peaks(filtered, height=threshold, distance=min_dist)
    return peaks.astype(np.int64)

@register_atom(witness_peak_correction)
@icontract.require(lambda filtered: isinstance(filtered, np.ndarray), "filtered must be a numpy array")
@icontract.require(lambda rpeaks: isinstance(rpeaks, np.ndarray), "rpeaks must be a numpy array")
@icontract.ensure(lambda result: result is not None, "Peak Correction output must not be None")
def peak_correction(filtered: np.ndarray, rpeaks: np.ndarray) -> np.ndarray:
    """Refine R-peak locations by snapping each detected peak to the nearest local maximum within a tolerance window, correcting for slight timing errors in the initial detection.

    Args:
        filtered: filtered electrocardiogram (ECG) signal
        rpeaks: initial R-peak indices

    Returns:
        corrected R-peak indices
    """
    # Snap each R-peak to the nearest local maximum within a tolerance window
    tol = 10  # samples
    corrected = np.empty_like(rpeaks)
    for i, pk in enumerate(rpeaks):
        lo = max(0, pk - tol)
        hi = min(len(filtered), pk + tol + 1)
        window = filtered[lo:hi]
        corrected[i] = lo + np.argmax(window)
    return corrected.astype(np.int64)

@register_atom(witness_template_extraction)
@icontract.require(lambda filtered: isinstance(filtered, np.ndarray), "filtered must be a numpy array")
@icontract.require(lambda rpeaks: isinstance(rpeaks, np.ndarray), "rpeaks must be a numpy array")
@icontract.ensure(lambda result: all(r is not None for r in result), "Template Extraction all outputs must not be None")
def template_extraction(filtered: np.ndarray, rpeaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract individual heartbeat waveform templates by slicing a window around each R-peak (the dominant spike in each heartbeat). Each template captures one full cardiac cycle for averaging or morphology analysis.

    Args:
        filtered: filtered electrocardiogram (ECG) signal
        rpeaks: corrected R-peak indices

    Returns:
        templates: 2D array of heartbeat templates
        rpeaks_final: final R-peak indices after template extraction
    """
    # Extract heartbeat windows around each R-peak
    # Window: 100 samples before, 150 samples after (typical for 360 Hz)
    pre, post = 100, 150
    window_len = pre + post
    valid_peaks = []
    templates_list = []
    for pk in rpeaks:
        lo = int(pk) - pre
        hi = int(pk) + post
        if lo >= 0 and hi <= len(filtered):
            templates_list.append(filtered[lo:hi])
            valid_peaks.append(pk)
    if len(templates_list) == 0:
        return (np.empty((0, window_len), dtype=filtered.dtype),
                np.empty(0, dtype=np.int64))
    templates = np.stack(templates_list, axis=0)
    rpeaks_final = np.array(valid_peaks, dtype=np.int64)
    return (templates, rpeaks_final)

@register_atom(witness_heart_rate_computation)
@icontract.require(lambda rpeaks: isinstance(rpeaks, np.ndarray), "rpeaks must be a numpy array")
@icontract.ensure(lambda result: all(r is not None for r in result), "Heart Rate Computation all outputs must not be None")
def heart_rate_computation(rpeaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous heart rate in bpm from R-R intervals with optional smoothing.

    Args:
        rpeaks: R-peak sample indices

    Returns:
        hr_idx: time indices for heart rate values
        heart_rate: instantaneous heart rate in bpm
    """
    if len(rpeaks) < 2:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
    # R-R intervals in samples; assume 360 Hz
    fs = 360.0
    rr_intervals = np.diff(rpeaks).astype(np.float64)
    # Instantaneous heart rate in bpm
    heart_rate = 60.0 * fs / rr_intervals
    # Time indices: midpoints between consecutive R-peaks
    hr_idx = rpeaks[:-1] + np.diff(rpeaks) // 2
    return (hr_idx.astype(np.int64), heart_rate)


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _bandpass_filter_ffi(signal):
    """Wrapper that calls the C++ version of bandpass filter. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./bandpass_filter.so")
    _func_name = 'bandpass_filter_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(signal)

def _r_peak_detection_ffi(filtered):
    """Wrapper that calls the C++ version of r-peak detection. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./r_peak_detection.so")
    _func_name = 'r_peak_detection_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filtered)

def _peak_correction_ffi(filtered, rpeaks):
    """Wrapper that calls the C++ version of peak correction. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./peak_correction.so")
    _func_name = 'peak_correction_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filtered, rpeaks)

def _template_extraction_ffi(filtered, rpeaks):
    """Wrapper that calls the C++ version of template extraction. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./template_extraction.so")
    _func_name = 'template_extraction_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filtered, rpeaks)

def _heart_rate_computation_ffi(rpeaks):
    """Wrapper that calls the C++ version of heart rate computation. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./heart_rate_computation.so")
    _func_name = 'heart_rate_computation_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(rpeaks)