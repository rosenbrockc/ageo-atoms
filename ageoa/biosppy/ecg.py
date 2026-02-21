"""Auto-generated stateful atom wrappers following the ageoa pattern."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

# Import the original class for __new__ instantiation
# from <source_module> import ECGProcessor

# State model should be imported from the generated state_models module
# from <state_module> import ECGPipelineState

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_bandpass_filter)
def bandpass_filter(signal: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Apply FIR bandpass filter (3-45 Hz) to remove baseline wander and high-frequency noise from the raw ECG signal
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
def r_peak_detection(filtered: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Detect R-peak locations in the filtered ECG signal using the Hamilton segmenter algorithm
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
def peak_correction(filtered: np.ndarray, rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Correct R-peak locations to the nearest local maximum within a tolerance window
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
def template_extraction(filtered: np.ndarray, rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Extract individual heartbeat waveform templates around each R-peak with configurable before/after windows
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
def heart_rate_computation(rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Compute instantaneous heart rate in bpm from R-R intervals with optional smoothing
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
