from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal, ANYTHING

def witness_bandpass_filter(signal: AbstractArray) -> AbstractArray:
    """Shape-and-type check for bandpass filter. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result

def witness_r_peak_detection(filtered: AbstractArray) -> AbstractArray:
    """Shape-and-type check for r-peak detection. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=filtered.shape,
        dtype="float64",
    )
    return result

def witness_peak_correction(filtered: AbstractArray, rpeaks: AbstractArray) -> AbstractArray:
    """Shape-and-type check for peak correction. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=rpeaks.shape,
        dtype="float64",
    )
    return result

def witness_template_extraction(filtered: AbstractArray, rpeaks: AbstractArray) -> AbstractArray:
    """Shape-and-type check for template extraction. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=filtered.shape,  # This is a simplification
        dtype="float64",
    )
    return result

def witness_heart_rate_computation(rpeaks: AbstractArray) -> AbstractArray:
    """Shape-and-type check for heart rate computation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=rpeaks.shape,
        dtype="float64",
    )
    return result
