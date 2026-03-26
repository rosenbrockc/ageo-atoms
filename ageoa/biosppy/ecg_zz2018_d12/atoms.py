from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_assemblezz2018sqi, witness_computebeatagreementsqi, witness_computefrequencysqi, witness_computekurtosissqi
from biosppy.signals.ecg import bSQI
from biosppy.signals.ecg import fSQI
from biosppy.signals.ecg import kSQI

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_computebeatagreementsqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.require(lambda search_window: isinstance(search_window, (float, int, np.number)), "search_window must be numeric")
@icontract.ensure(lambda result: result is not None, "ComputeBeatAgreementSQI output must not be None")
def computebeatagreementsqi(detector_1: object, detector_2: object, fs: float, mode: str, search_window: float) -> float:
    """Computes a beat-detector agreement Signal Quality Index (SQI) for an electrocardiogram (ECG) signal. Compares heartbeat locations found by two independent detectors: if they agree on where beats occur (within a search window), the signal is likely clean.

    Args:
        detector_1: beat locations from the first detector, same timeline as detector_2
        detector_2: beat locations from the second detector, same timeline as detector_1
        fs: sampling rate > 0
        mode: valid bSQI computation mode
        search_window: window >= 0

    Returns:
        normalized detector agreement score (0 = no agreement, 1 = full agreement)
    """
    return bSQI(detector_1=detector_1, detector_2=detector_2, fs=fs, mode=mode, search_window=search_window)

@register_atom(witness_computefrequencysqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "ComputeFrequencySQI output must not be None")
def computefrequencysqi(ecg_signal: object, fs: float, nseg: int, num_spectrum: object, dem_spectrum: object, mode: str) -> float:
    """Computes a frequency-domain Signal Quality Index (SQI) for an electrocardiogram (ECG) signal. Measures the ratio of signal power in expected ECG frequency bands versus total or noise bands — a clean ECG concentrates power in narrow physiological frequency ranges.

    Args:
        ecg_signal: 1-D sampled ECG
        fs: sampling rate > 0
        nseg: segment length > 0
        num_spectrum: numerator frequency range (expected ECG power band)
        dem_spectrum: denominator frequency range (reference or total band)
        mode: valid fSQI computation mode

    Returns:
        spectral quality ratio/score
    """
    return fSQI(ecg_signal=ecg_signal, fs=fs, nseg=nseg, num_spectrum=num_spectrum, dem_spectrum=dem_spectrum, mode=mode)

@register_atom(witness_computekurtosissqi)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda fisher: fisher is not None, "fisher cannot be None")
@icontract.ensure(lambda result: result is not None, "ComputeKurtosisSQI output must not be None")
def computekurtosissqi(signal: object, fisher: bool) -> float:
    """Computes a kurtosis-based Signal Quality Index (SQI) for an electrocardiogram (ECG) signal. Kurtosis measures how "peaked" or "tailed" a distribution is — clean ECG signals have characteristic kurtosis values due to their sharp R-peaks, while noisy signals deviate.

    Args:
        signal: 1-D sampled ECG
        fisher: if True, uses Fisher's definition (normal = 0); if False, uses Pearson's (normal = 3)

    Returns:
        kurtosis-derived quality score
    """
    return kSQI(signal=signal, fisher=fisher)

@register_atom(witness_assemblezz2018sqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.require(lambda search_window: isinstance(search_window, (float, int, np.number)), "search_window must be numeric")
@icontract.require(lambda b_sqi: isinstance(b_sqi, (float, int, np.number)), "b_sqi must be numeric")
@icontract.require(lambda f_sqi: isinstance(f_sqi, (float, int, np.number)), "f_sqi must be numeric")
@icontract.require(lambda k_sqi: isinstance(k_sqi, (float, int, np.number)), "k_sqi must be numeric")
@icontract.ensure(lambda result: result is not None, "AssembleZZ2018SQI output must not be None")
def assemblezz2018sqi(signal: object, detector_1: object, detector_2: object, fs: float, search_window: float, nseg: int, mode: str, b_sqi: float, f_sqi: float, k_sqi: float) -> object:
    """Builds the final signal-quality score by combining beat-agreement, frequency, and shape evidence.

    Args:
        signal: 1-D sampled heart signal
        detector_1: first beat detector, aligned with signal
        detector_2: second beat detector, aligned with signal
        fs: sampling rate (> 0)
        search_window: search window size (>= 0)
        nseg: segment length (> 0)
        mode: processing mode
        b_sqi: beat-agreement quality score
        f_sqi: frequency quality score
        k_sqi: shape quality score

    Returns:
        final composite quality output
    """
    return {"b_sqi": b_sqi, "f_sqi": f_sqi, "k_sqi": k_sqi}
