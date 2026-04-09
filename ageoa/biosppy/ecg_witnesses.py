"""Ghost witnesses for ECG atoms."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractArray, AbstractSignal
except ImportError:
    pass


def witness_bandpass_filter(
    signal: AbstractSignal,
    sampling_rate: AbstractArray,
) -> AbstractSignal:
    """Ghost witness for Bandpass Filter."""
    return AbstractSignal(
        shape=signal.shape,
        dtype=signal.dtype,
        sampling_rate=getattr(signal, "sampling_rate", 1000.0),
        domain=getattr(signal, "domain", "time"),
    )


def witness_r_peak_detection(
    filtered: AbstractArray,
    sampling_rate: AbstractArray,
) -> AbstractArray:
    """Ghost witness for R-Peak Detection."""
    return AbstractArray(shape=filtered.shape, dtype=filtered.dtype)


def witness_peak_correction(
    filtered: AbstractArray,
    rpeaks: AbstractArray,
    sampling_rate: AbstractArray,
) -> AbstractArray:
    """Ghost witness for Peak Correction."""
    return AbstractArray(shape=rpeaks.shape, dtype=rpeaks.dtype)


def witness_reject_outlier_intervals(
    rpeaks: AbstractArray,
    sampling_rate: AbstractArray,
) -> AbstractArray:
    """Ghost witness for outlier interval rejection."""
    return AbstractArray(shape=rpeaks.shape, dtype=rpeaks.dtype)


def witness_template_extraction(
    filtered: AbstractArray,
    rpeaks: AbstractArray,
    sampling_rate: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for Template Extraction."""
    return (
        AbstractArray(shape=filtered.shape, dtype=filtered.dtype),
        AbstractArray(shape=rpeaks.shape, dtype=rpeaks.dtype),
    )


def witness_heart_rate_computation(
    rpeaks: AbstractArray,
    sampling_rate: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for Heart Rate Computation."""
    return (
        AbstractArray(shape=rpeaks.shape, dtype=rpeaks.dtype),
        AbstractArray(shape=rpeaks.shape, dtype=rpeaks.dtype),
    )


def witness_ssf_segmenter(
    signal: AbstractSignal,
    sampling_rate: AbstractArray,
) -> AbstractArray:
    """Ghost witness for SSF Segmenter."""
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)


def witness_christov_segmenter(
    signal: AbstractSignal,
    sampling_rate: AbstractArray,
) -> AbstractArray:
    """Ghost witness for Christov Segmenter."""
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)
