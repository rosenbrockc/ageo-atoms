from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_kazemi_peak_detection(
    signal: AbstractArray,
    sampling_freq: AbstractScalar,
    seconds: AbstractScalar,
    overlap: AbstractScalar,
    minlen: AbstractScalar,
) -> AbstractArray:
    """Shape-and-type check for kazemi peak detection."""
    del sampling_freq, seconds, overlap, minlen
    return AbstractArray(shape=(signal.shape[0],), dtype="intp")


def witness_ppg_reconstruction(
    sig: AbstractArray,
    clean_indices: AbstractArray,
    noisy_indices: AbstractArray,
    sampling_rate: AbstractScalar,
    filter_signal: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Shape-and-type check for ppg reconstruction."""
    del clean_indices, noisy_indices, sampling_rate, filter_signal
    index_meta = AbstractArray(shape=(sig.shape[0],), dtype="intp")
    return AbstractArray(shape=sig.shape, dtype=sig.dtype), index_meta, index_meta


def witness_ppg_sqa(
    sig: AbstractArray,
    sampling_rate: AbstractScalar,
    filter_signal: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for ppg sqa."""
    del sampling_rate, filter_signal
    index_meta = AbstractArray(shape=(sig.shape[0],), dtype="intp")
    return index_meta, index_meta
