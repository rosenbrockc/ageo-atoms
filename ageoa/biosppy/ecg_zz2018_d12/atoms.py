from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_assemblezz2018sqi, witness_computebeatagreementsqi, witness_computefrequencysqi, witness_computekurtosissqi

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_computebeatagreementsqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.require(lambda search_window: isinstance(search_window, (float, int, np.number)), "search_window must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "ComputeBeatAgreementSQI output must not be None")
def computebeatagreementsqi(detector_1: object, detector_2: object, fs: float, mode: str, search_window: float) -> float:
    """Computes a beat-detector agreement quality index from two detector streams within a temporal search window.

    Args:
        detector_1: same timeline as detector_2
        detector_2: same timeline as detector_1
        fs: sampling rate > 0
        mode: valid bSQI computation mode
        search_window: window >= 0

    Returns:
        normalized detector agreement score
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_computefrequencysqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "ComputeFrequencySQI output must not be None")
def computefrequencysqi(ecg_signal: object, fs: float, nseg: int, num_spectrum: object, dem_spectrum: object, mode: str) -> float:
    """Computes a spectral quality index from ECG signal power in numerator and denominator spectral bands.

    Args:
        ecg_signal: 1-D sampled ECG
        fs: sampling rate > 0
        nseg: segment length > 0
        num_spectrum: valid frequency range
        dem_spectrum: valid frequency range
        mode: valid fSQI computation mode

    Returns:
        spectral quality ratio/score
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_computekurtosissqi)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda fisher: fisher is not None, "fisher cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "ComputeKurtosisSQI output must not be None")
def computekurtosissqi(signal: object, fisher: bool) -> float:
    """Computes kurtosis-based ECG quality score from the input signal.

    Args:
        signal: 1-D sampled ECG
        fisher: selects Fisher vs Pearson convention

    Returns:
        kurtosis-derived quality score
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_assemblezz2018sqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.require(lambda search_window: isinstance(search_window, (float, int, np.number)), "search_window must be numeric")
@icontract.require(lambda b_sqi: isinstance(b_sqi, (float, int, np.number)), "b_sqi must be numeric")
@icontract.require(lambda f_sqi: isinstance(f_sqi, (float, int, np.number)), "f_sqi must be numeric")
@icontract.require(lambda k_sqi: isinstance(k_sqi, (float, int, np.number)), "k_sqi must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "AssembleZZ2018SQI output must not be None")
def assemblezz2018sqi(signal: object, detector_1: object, detector_2: object, fs: float, search_window: float, nseg: int, mode: str, b_sqi: float, f_sqi: float, k_sqi: float) -> object:
    """Builds the final ZZ2018 signal-quality result by combining beat, frequency, and kurtosis evidence.
def assemblezz2018sqi(signal: object, detector_1: object, detector_2: object, fs: float, search_window: float, nseg: int, mode: str, b_sqi: float, f_sqi: float, k_sqi: float) -> object:
    Args:
        signal: 1-D sampled ECG
        detector_1: aligned with signal
        detector_2: aligned with signal
        fs: sampling rate > 0
        search_window: window >= 0
        nseg: segment length > 0
        mode: valid ZZ2018 mode
        b_sqi: from ComputeBeatAgreementSQI
        f_sqi: from ComputeFrequencySQI
        k_sqi: from ComputeKurtosisSQI

    Returns:
        final composite quality output
    """
    raise NotImplementedError("Wire to original implementation")
