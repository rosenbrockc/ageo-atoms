"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_bandpass_filter)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.ensure(lambda result: result is not None, "Bandpass Filter output must not be None")
def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """Apply FIR bandpass filter (3-45 Hz) to remove baseline wander and high-frequency noise from the raw ECG signal.

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
    """Detect R-peak locations in the filtered ECG signal using the Hamilton segmenter algorithm.

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
    """Correct R-peak locations to the nearest local maximum within a tolerance window.

    Args:
        filtered: filtered ECG signal
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
    """Extract individual heartbeat waveform templates around each R-peak with configurable before/after windows.

    Args:
        filtered: filtered ECG signal
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

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def _bandpass_filter_ffi(signal):
    """FFI bridge to C++ implementation of Bandpass Filter."""
    _lib = ctypes.CDLL("./bandpass_filter.so")
    _func_name = atom.method_names[0] if atom.method_names else 'bandpass_filter'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(signal)

def _r_peak_detection_ffi(filtered):
    """FFI bridge to C++ implementation of R-Peak Detection."""
    _lib = ctypes.CDLL("./r_peak_detection.so")
    _func_name = atom.method_names[0] if atom.method_names else 'r_peak_detection'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filtered)

def _peak_correction_ffi(filtered, rpeaks):
    """FFI bridge to C++ implementation of Peak Correction."""
    _lib = ctypes.CDLL("./peak_correction.so")
    _func_name = atom.method_names[0] if atom.method_names else 'peak_correction'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filtered, rpeaks)

def _template_extraction_ffi(filtered, rpeaks):
    """FFI bridge to C++ implementation of Template Extraction."""
    _lib = ctypes.CDLL("./template_extraction.so")
    _func_name = atom.method_names[0] if atom.method_names else 'template_extraction'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filtered, rpeaks)

def _heart_rate_computation_ffi(rpeaks):
    """FFI bridge to C++ implementation of Heart Rate Computation."""
    _lib = ctypes.CDLL("./heart_rate_computation.so")
    _func_name = atom.method_names[0] if atom.method_names else 'heart_rate_computation'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(rpeaks)
