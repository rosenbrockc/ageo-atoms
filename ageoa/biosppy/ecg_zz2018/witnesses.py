from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_calculatecompositesqi_zz2018(signal: AbstractSignal,
    detector_1: AbstractSignal,
    detector_2: AbstractSignal,
    fs: AbstractScalar,
    search_window: AbstractScalar,
    nseg: AbstractScalar,
    mode: AbstractScalar,
) -> AbstractArray:
    """Ghost witness for CalculateCompositeSQI_ZZ2018."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result

def witness_calculatebeatagreementsqi(detector_1: AbstractSignal, detector_2: AbstractSignal, fs: AbstractSignal, mode: AbstractSignal, search_window: AbstractSignal) -> AbstractSignal:
    """Ghost witness for CalculateBeatAgreementSQI."""
    result = AbstractSignal(
        shape=detector_1.shape,
        dtype="float64",
        sampling_rate=getattr(detector_1, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

def witness_calculatefrequencypowersqi(ecg_signal: AbstractSignal, fs: AbstractSignal, nseg: AbstractSignal, num_spectrum: AbstractSignal, dem_spectrum: AbstractSignal, mode: AbstractSignal) -> AbstractSignal:
    """Ghost witness for CalculateFrequencyPowerSQI."""
    result = AbstractSignal(
        shape=ecg_signal.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

def witness_calculatekurtosissqi(signal: AbstractSignal, fisher: AbstractSignal) -> AbstractSignal:
    """Ghost witness for CalculateKurtosisSQI."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
