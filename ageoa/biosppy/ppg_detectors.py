from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .ppg_detectors_witnesses import witness_detect_signal_onsets_elgendi2013
from biosppy.signals.ppg import find_onsets_elgendi2013

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_detect_signal_onsets_elgendi2013)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda peakwindow: isinstance(peakwindow, (float, int, np.number)), "peakwindow must be numeric")
@icontract.require(lambda beatwindow: isinstance(beatwindow, (float, int, np.number)), "beatwindow must be numeric")
@icontract.require(lambda beatoffset: isinstance(beatoffset, (float, int, np.number)), "beatoffset must be numeric")
@icontract.require(lambda mindelay: isinstance(mindelay, (float, int, np.number)), "mindelay must be numeric")
@icontract.ensure(lambda result: result is not None, "detect_signal_onsets_elgendi2013 output must not be None")
def detect_signal_onsets_elgendi2013(signal: np.ndarray, sampling_rate: float, peakwindow: float, beatwindow: float, beatoffset: float, mindelay: float) -> np.ndarray:
    """Detects physiological signal onsets (e.g., electrocardiogram (ECG) R-peaks) using the Elgendi et al. (2013) method. This algorithm typically involves bandpass filtering, squaring, generating potential blocks of interest, and applying dynamic thresholding to identify definitive peaks.

Args:
    signal: 1D numerical array representing the physiological signal.
    sampling_rate: The sampling frequency of the signal in Hz.
    peakwindow: Duration of the peak detection window in seconds, used for identifying blocks of interest.
    beatwindow: Duration of the beat classification window in seconds, used for dynamic thresholding.
    beatoffset: Duration of the offset for the beat classification window in seconds.
    mindelay: Minimum delay between consecutive detected onsets in seconds.

Returns:
    An array of integer indices corresponding to the detected onsets within the input signal."""
    return find_onsets_elgendi2013(signal=signal, sampling_rate=sampling_rate, peakwindow=peakwindow, beatwindow=beatwindow, beatoffset=beatoffset, mindelay=mindelay)["onsets"]

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .ppg_detectors_witnesses import witness_detectonsetevents
from biosppy.signals.ppg import find_onsets_kavsaoglu2016

@register_atom(witness_detectonsetevents)
@icontract.require(lambda signal: signal.ndim >= 1, "signal must be at least 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be np.ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda alpha: isinstance(alpha, (float, int, np.number)), "alpha must be numeric")
@icontract.require(lambda init_bpm: isinstance(init_bpm, (float, int, np.number)), "init_bpm must be numeric")
@icontract.require(lambda min_delay: isinstance(min_delay, (float, int, np.number)), "min_delay must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def detectonsetevents(signal: np.ndarray, sampling_rate: float, alpha: float, k: int, init_bpm: float, min_delay: float, max_BPM: float) -> np.ndarray:
    """Detect rhythmic onset events from an input signal using provided tempo and delay constraints.

Args:
    signal: 1-D sampled signal array.
    sampling_rate: Sampling frequency in Hz; must be > 0.
    alpha: Algorithm coefficient.
    k: Window/order parameter; typically > 0.
    init_bpm: Initial tempo estimate in beats per minute (BPM); must be > 0.
    min_delay: Minimum inter-onset delay; must be >= 0.
    max_BPM: Upper tempo bound in BPM; must be > 0.

Returns:
    Detected onset locations/times; may be empty."""
    return find_onsets_kavsaoglu2016(signal=signal, sampling_rate=sampling_rate, alpha=alpha, k=k, init_bpm=init_bpm, min_delay=min_delay, max_BPM=max_BPM)["onsets"]
