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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")


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