from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal, ANYTHING

def witness_bandpass_filter(signal: AbstractArray) -> AbstractArray:
    """Ghost witness for Bandpass Filter."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result

def witness_r_peak_detection(filtered: AbstractArray) -> AbstractArray:
    """Ghost witness for R-Peak Detection."""
    result = AbstractArray(
        shape=filtered.shape,
        dtype="float64",
    )
    return result

def witness_peak_correction(filtered: AbstractArray, rpeaks: AbstractArray) -> AbstractArray:
    """Ghost witness for Peak Correction."""
    result = AbstractArray(
        shape=rpeaks.shape,
        dtype="float64",
    )
    return result

def witness_template_extraction(filtered: AbstractArray, rpeaks: AbstractArray) -> AbstractArray:
    """Ghost witness for Template Extraction."""
    result = AbstractArray(
        shape=filtered.shape,  # This is a simplification
        dtype="float64",
    )
    return result

def witness_heart_rate_computation(rpeaks: AbstractArray) -> AbstractArray:
    """Ghost witness for Heart Rate Computation."""
    result = AbstractArray(
        shape=rpeaks.shape,
        dtype="float64",
    )
    return result
