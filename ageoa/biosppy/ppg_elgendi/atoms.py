from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_detect_signal_onsets_elgendi2013

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
    raise NotImplementedError("Wire to original implementation")
