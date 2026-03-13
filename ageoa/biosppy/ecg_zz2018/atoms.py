from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_calculatebeatagreementsqi, witness_calculatecompositesqi_zz2018, witness_calculatefrequencypowersqi, witness_calculatekurtosissqi

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_calculatecompositesqi_zz2018)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateCompositeSQI_ZZ2018 output must not be None")
def calculatecompositesqi_zz2018(signal: NDArray, detector_1: NDArray, detector_2: NDArray, fs: float, search_window: int, nseg: int, mode: str) -> float:
    """Calculates a composite Signal Quality Index (SQI) for a signal, using multiple detectors and parameters. This likely serves as an orchestrator or a specific implementation from a paper.

    Args:
        signal: Primary physiological signal waveform.
        detector_1: Array of beat detections from the first detector.
        detector_2: Array of beat detections from the second detector.
        fs: Sampling frequency of the signal.
        search_window: Window size for searching or comparison.
        nseg: Number of segments for spectral analysis.
        mode: Operational mode for the calculation.

    Returns:
        The final composite SQI score.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_calculatebeatagreementsqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateBeatAgreementSQI output must not be None")
def calculatebeatagreementsqi(detector_1: NDArray, detector_2: NDArray, fs: float, mode: str, search_window: int) -> float:
    """Calculates a beat-based Signal Quality Index (bSQI) based on the agreement between two beat detectors.

    Args:
        detector_1: Array of beat detections from the first detector.
        detector_2: Array of beat detections from the second detector.
        fs: Sampling frequency of the signal.
        mode: Operational mode for the calculation.
        search_window: Window size for comparing detector outputs.

    Returns:
        The beat agreement SQI score.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_calculatefrequencypowersqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateFrequencyPowerSQI output must not be None")
def calculatefrequencypowersqi(ecg_signal: NDArray, fs: float, nseg: int, num_spectrum: NDArray, dem_spectrum: NDArray, mode: str) -> float:
    """Calculates a frequency-based Signal Quality Index (fSQI) using the power spectrum of the ECG signal.

    Args:
        ecg_signal: The ECG signal waveform.
        fs: Sampling frequency of the signal.
        nseg: Number of segments for spectral analysis.
        num_spectrum: Numerator of the spectral ratio.
        dem_spectrum: Denominator of the spectral ratio.
        mode: Operational mode for the calculation.

    Returns:
        The frequency power SQI score.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_calculatekurtosissqi)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda fisher: fisher is not None, "fisher cannot be None")
@icontract.ensure(lambda result: result is not None, "CalculateKurtosisSQI output must not be None")
def calculatekurtosissqi(signal: NDArray, fisher: bool) -> float:
    """Calculates a Signal Quality Index (kSQI) based on the statistical kurtosis of the signal.

    Args:
        signal: The input signal waveform.
        fisher: Flag to indicate if Fisher_primes definition of kurtosis is used.

    Returns:
        The kurtosis-based SQI score.
    """
    raise NotImplementedError("Wire to original implementation")
