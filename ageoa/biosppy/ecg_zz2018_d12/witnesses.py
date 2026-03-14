from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_computebeatagreementsqi(detector_1: AbstractArray, detector_2: AbstractArray, fs: AbstractArray, mode: AbstractArray, search_window: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute beat agreement sqi. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=detector_1.shape,
        dtype="float64",
    )
    return result

def witness_computefrequencysqi(ecg_signal: AbstractSignal, fs: AbstractSignal, nseg: AbstractSignal, num_spectrum: AbstractSignal, dem_spectrum: AbstractSignal, mode: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for compute frequency sqi. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=ecg_signal.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

def witness_computekurtosissqi(signal: AbstractArray, fisher: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute kurtosis sqi. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result

def witness_assemblezz2018sqi(signal: AbstractArray, detector_1: AbstractArray, detector_2: AbstractArray, fs: AbstractArray, search_window: AbstractArray, nseg: AbstractArray, mode: AbstractArray, b_sqi: AbstractArray, f_sqi: AbstractArray, k_sqi: AbstractArray) -> AbstractArray:
    """Shape-and-type check for assemble zz2018 sqi. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result
