"""Auto-generated stateful atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

# Import the original class for __new__ instantiation
from .ecg_processor import ECGProcessor

# State model should be imported from the generated state_models module
from .ecg_state import ECGPipelineState

from .ecg_witnesses import witness_bandpass_filter, witness_r_peak_detection, witness_peak_correction, witness_template_extraction, witness_heart_rate_computation

@register_atom(witness_bandpass_filter)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.ensure(lambda result, **kwargs: result is not None, "Bandpass Filter output must not be None")
def bandpass_filter(signal: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Apply FIR bandpass filter (3-45 Hz) to remove baseline wander and high-frequency noise from the raw ECG signal

    Args:
        signal: 1D raw ECG signal
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[bandpass-filtered ECG, ECGPipelineState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    obj.rpeaks = state.rpeaks
    obj.filter_signal(signal)
    new_state = state.model_copy(update={
        "filtered": obj.filtered,
        "rpeaks": obj.rpeaks,
    })
    result = obj.filtered
    return result, new_state

@register_atom(witness_r_peak_detection)
@icontract.require(lambda filtered: isinstance(filtered, np.ndarray), "filtered must be a numpy array")
@icontract.ensure(lambda result, **kwargs: result is not None, "R-Peak Detection output must not be None")
def r_peak_detection(filtered: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Detect R-peak locations in the filtered ECG signal using the Hamilton segmenter algorithm

    Args:
        filtered: filtered ECG signal
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[R-peak sample indices, ECGPipelineState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    obj.rpeaks = state.rpeaks
    obj.detect_rpeaks(filtered)
    new_state = state.model_copy(update={
        "filtered": obj.filtered,
        "rpeaks": obj.rpeaks,
    })
    result = obj.rpeaks
    return result, new_state

@register_atom(witness_peak_correction)
@icontract.require(lambda filtered: isinstance(filtered, np.ndarray), "filtered must be a numpy array")
@icontract.require(lambda rpeaks: isinstance(rpeaks, np.ndarray), "rpeaks must be a numpy array")
@icontract.ensure(lambda result, **kwargs: result is not None, "Peak Correction output must not be None")
def peak_correction(filtered: np.ndarray, rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Correct R-peak locations to the nearest local maximum within a tolerance window

    Args:
        filtered: filtered ECG signal
        rpeaks: initial R-peak indices
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[corrected R-peak indices, ECGPipelineState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    obj.rpeaks = state.rpeaks
    obj.correct_peaks(filtered, rpeaks)
    new_state = state.model_copy(update={
        "filtered": obj.filtered,
        "rpeaks": obj.rpeaks,
    })
    result = obj.rpeaks_corrected
    return result, new_state

@register_atom(witness_template_extraction)
@icontract.require(lambda filtered: isinstance(filtered, np.ndarray), "filtered must be a numpy array")
@icontract.require(lambda rpeaks: isinstance(rpeaks, np.ndarray), "rpeaks must be a numpy array")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "Template Extraction all outputs must not be None")
def template_extraction(filtered: np.ndarray, rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Extract individual heartbeat waveform templates around each R-peak with configurable before/after windows

    Args:
        filtered: filtered ECG signal
        rpeaks: corrected R-peak indices
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[tuple[templates, rpeaks_final], ECGPipelineState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    obj.rpeaks = state.rpeaks
    obj.extract_templates(filtered, rpeaks)
    new_state = state.model_copy(update={
        "filtered": obj.filtered,
        "rpeaks": obj.rpeaks,
    })
    result = (obj.templates, obj.rpeaks_final)
    return result, new_state

@register_atom(witness_heart_rate_computation)
@icontract.require(lambda rpeaks: isinstance(rpeaks, np.ndarray), "rpeaks must be a numpy array")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "Heart Rate Computation all outputs must not be None")
def heart_rate_computation(rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Compute instantaneous heart rate in bpm from R-R intervals with optional smoothing

    Args:
        rpeaks: R-peak sample indices
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[tuple[hr_idx, heart_rate], ECGPipelineState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    obj.rpeaks = state.rpeaks
    obj.compute_heart_rate(rpeaks)
    new_state = state.model_copy(update={
        "filtered": obj.filtered,
        "rpeaks": obj.rpeaks,
    })
    result = (obj.hr_idx, obj.heart_rate)
    return result, new_state
