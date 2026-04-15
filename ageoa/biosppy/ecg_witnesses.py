"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import sciona.ghost.abstract as _ghost_abstract
except ImportError:
    _ghost_abstract = None

@dataclass
class _ScionaGhostArrayFallback:
    shape: tuple = ()
    dtype: str = "float64"

@dataclass
class _ScionaGhostSignalFallback(_ScionaGhostArrayFallback):
    sampling_rate: float = 44100.0
    domain: str = "time"

@dataclass
class _ScionaGhostScalarFallback:
    dtype: str = "float64"

AbstractArray = getattr(_ghost_abstract, "AbstractArray", _ScionaGhostArrayFallback)
AbstractSignal = getattr(_ghost_abstract, "AbstractSignal", _ScionaGhostSignalFallback)
AbstractScalar = getattr(_ghost_abstract, "AbstractScalar", _ScionaGhostScalarFallback)


def witness_bandpass_filter(signal: AbstractSignal, state: AbstractArray) -> tuple[AbstractSignal, AbstractArray]:
    """Ghost witness for Bandpass Filter (unknown, state-updating)."""
    result = AbstractSignal(shape=signal.shape, dtype=signal.dtype, sampling_rate=getattr(signal, 'sampling_rate', 44100.0), domain=getattr(signal, 'domain', 'time'))
    return result, state

def witness_r_peak_detection(filtered: AbstractArray, state: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for R-Peak Detection (unknown, state-updating)."""
    result = AbstractArray(shape=filtered.shape, dtype=filtered.dtype)
    return result, state

def witness_peak_correction(filtered: AbstractArray, rpeaks: AbstractArray, state: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for Peak Correction (unknown, state-updating)."""
    result = AbstractArray(shape=filtered.shape, dtype=filtered.dtype)
    return result, state

def witness_template_extraction(filtered: AbstractArray, rpeaks: AbstractArray, state: AbstractArray) -> tuple[tuple[AbstractArray, AbstractArray], AbstractArray]:
    """Ghost witness for Template Extraction (unknown, state-updating)."""
    result = (
        AbstractArray(shape=filtered.shape, dtype=filtered.dtype),
        AbstractArray(shape=filtered.shape, dtype=filtered.dtype),
    )
    return result, state

def witness_heart_rate_computation(rpeaks: AbstractArray, state: AbstractArray) -> tuple[tuple[AbstractArray, AbstractArray], AbstractArray]:
    """Ghost witness for Heart Rate Computation (unknown, state-updating)."""
    result = (
        AbstractArray(shape=rpeaks.shape, dtype=rpeaks.dtype),
        AbstractArray(shape=rpeaks.shape, dtype=rpeaks.dtype),
    )
    return result, state
