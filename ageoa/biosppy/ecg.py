"""Auto-generated stateful atom wrappers following the sciona pattern."""

from __future__ import annotations

# mypy: disable-error-code=untyped-decorator

from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import icontract
from sciona.ghost.registry import register_atom

_SCIONA_UNSET = object()

import importlib.util

_SCIONA_SOURCE_FILE = '/private/var/folders/yt/f_48fkt53wbdg3hpw33h5cd40000gn/T/pytest-of-conrad/pytest-809/test_write_artefacts0/ecg_processor.py'
_SCIONA_SOURCE_SPEC = importlib.util.spec_from_file_location("_sciona_ingest_source", _SCIONA_SOURCE_FILE)
if _SCIONA_SOURCE_SPEC is None or _SCIONA_SOURCE_SPEC.loader is None:
    raise ImportError(f"Unable to load source module from {_SCIONA_SOURCE_FILE}")
_SCIONA_SOURCE_MODULE = importlib.util.module_from_spec(_SCIONA_SOURCE_SPEC)
_SCIONA_SOURCE_SPEC.loader.exec_module(_SCIONA_SOURCE_MODULE)
ECGProcessor: Any = getattr(_SCIONA_SOURCE_MODULE, "ECGProcessor")
_SCIONA_SOURCE_SYMBOL: Any = ECGProcessor

from state_models import ECGPipelineState

from witnesses import witness_bandpass_filter, witness_heart_rate_computation, witness_peak_correction, witness_r_peak_detection, witness_template_extraction

@register_atom(witness_bandpass_filter)
@icontract.require(lambda signal: hasattr(signal, "__array__") or isinstance(signal, (np.ndarray, list, tuple)), "signal must be array-like")
@icontract.ensure(lambda result: result is not None, "Bandpass Filter output must not be None")
def bandpass_filter(signal: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Apply FIR bandpass filter (3-45 Hz) to remove baseline wander and high-frequency noise from the raw ECG signal

    Args:
        signal: 1D raw ECG signal
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[np.ndarray, ECGPipelineState]
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    raise NotImplementedError("Bandpass Filter: no canonical source for required state slot 'sampling_rate'")

@register_atom(witness_r_peak_detection)
@icontract.require(lambda filtered: hasattr(filtered, "__array__") or isinstance(filtered, (np.ndarray, list, tuple)), "filtered must be array-like")
@icontract.ensure(lambda result: result is not None, "R-Peak Detection output must not be None")
def r_peak_detection(filtered: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Detect R-peak locations in the filtered ECG signal using the Hamilton segmenter algorithm

    Args:
        filtered: filtered ECG signal
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[np.ndarray, ECGPipelineState]
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    raise NotImplementedError("R-Peak Detection: no canonical source for required state slot 'sampling_rate'")

@register_atom(witness_peak_correction)
@icontract.require(lambda filtered: hasattr(filtered, "__array__") or isinstance(filtered, (np.ndarray, list, tuple)), "filtered must be array-like")
@icontract.require(lambda rpeaks: hasattr(rpeaks, "__array__") or isinstance(rpeaks, (np.ndarray, list, tuple)), "rpeaks must be array-like")
@icontract.ensure(lambda result: result is not None, "Peak Correction output must not be None")
def peak_correction(filtered: np.ndarray, rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
    """Correct R-peak locations to the nearest local maximum within a tolerance window

    Args:
        filtered: filtered ECG signal
        rpeaks: initial R-peak indices
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[np.ndarray, ECGPipelineState]
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    obj.rpeaks = state.rpeaks
    raise NotImplementedError("Peak Correction: no canonical source for required state slot 'sampling_rate'")

@register_atom(witness_template_extraction)
@icontract.require(lambda filtered: hasattr(filtered, "__array__") or isinstance(filtered, (np.ndarray, list, tuple)), "filtered must be array-like")
@icontract.require(lambda rpeaks: hasattr(rpeaks, "__array__") or isinstance(rpeaks, (np.ndarray, list, tuple)), "rpeaks must be array-like")
@icontract.ensure(lambda result: all(r is not None for r in result), "Template Extraction all outputs must not be None")
def template_extraction(filtered: np.ndarray, rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]:
    """Extract individual heartbeat waveform templates around each R-peak with configurable before/after windows

    Args:
        filtered: filtered ECG signal
        rpeaks: corrected R-peak indices
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.filtered = state.filtered
    obj.rpeaks = state.rpeaks
    raise NotImplementedError("Template Extraction: no canonical source for required state slot 'sampling_rate'")

@register_atom(witness_heart_rate_computation)
@icontract.require(lambda rpeaks: hasattr(rpeaks, "__array__") or isinstance(rpeaks, (np.ndarray, list, tuple)), "rpeaks must be array-like")
@icontract.ensure(lambda result: all(r is not None for r in result), "Heart Rate Computation all outputs must not be None")
def heart_rate_computation(rpeaks: np.ndarray, state: ECGPipelineState) -> tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]:
    """Compute instantaneous heart rate in bpm from R-R intervals with optional smoothing

    Args:
        rpeaks: R-peak sample indices
        state: ECGPipelineState object containing cross-window persistent state.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], ECGPipelineState]
    """
    obj = ECGProcessor.__new__(ECGProcessor)
    obj.rpeaks = state.rpeaks
    raise NotImplementedError("Heart Rate Computation: no canonical source for required state slot 'sampling_rate'")
