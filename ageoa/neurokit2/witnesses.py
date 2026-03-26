from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_zhao2018hrvanalysis(ecg_cleaned: AbstractSignal, rpeaks: AbstractSignal, sampling_rate: AbstractSignal, window: AbstractSignal, mode: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for zhao2018 hrv analysis. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=ecg_cleaned.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_cleaned, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_averageqrstemplate(ecg_cleaned: AbstractSignal, rpeaks: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for average qrs template. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=ecg_cleaned.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_cleaned, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
