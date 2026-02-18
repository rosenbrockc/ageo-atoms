"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_bandpass_filter(signal: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Bandpass Filter."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate', 44100.0),
        domain="time",
    )

def witness_r_peak_detection(filtered: AbstractSignal) -> AbstractSignal:
    """Ghost witness for R-Peak Detection."""
    return AbstractSignal(
        shape=filtered.shape,
        dtype="float64",
        sampling_rate=getattr(filtered, 'sampling_rate', 44100.0),
        domain="time",
    )

def witness_peak_correction(filtered: AbstractSignal, rpeaks: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Peak Correction."""
    return AbstractSignal(
        shape=filtered.shape,
        dtype="float64",
        sampling_rate=getattr(filtered, 'sampling_rate', 44100.0),
        domain="time",
    )

def witness_template_extraction(filtered: AbstractSignal, rpeaks: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Template Extraction."""
    return AbstractSignal(
        shape=filtered.shape,
        dtype="float64",
        sampling_rate=getattr(filtered, 'sampling_rate', 44100.0),
        domain="time",
    )

def witness_heart_rate_computation(rpeaks: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Heart Rate Computation."""
    return AbstractSignal(
        shape=rpeaks.shape,
        dtype="float64",
        sampling_rate=getattr(rpeaks, 'sampling_rate', 44100.0),
        domain="time",
    )
