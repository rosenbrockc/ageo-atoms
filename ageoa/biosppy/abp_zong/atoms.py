from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np; from numpy.typing import NDArray

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from .witnesses import witness_audio_onset_detection
from biosppy.signals.abp import find_onsets_zong2003

@register_atom(witness_audio_onset_detection)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda sampling_rate: sampling_rate is not None, "sampling_rate cannot be None")
@icontract.require(lambda sm_size: sm_size is not None, "sm_size cannot be None")
@icontract.require(lambda size: size is not None, "size cannot be None")
@icontract.require(lambda alpha: alpha is not None, "alpha cannot be None")
@icontract.require(lambda wrange: wrange is not None, "wrange cannot be None")
@icontract.require(lambda d1_th: d1_th is not None, "d1_th cannot be None")
@icontract.require(lambda d2_th: d2_th is not None, "d2_th cannot be None")
@icontract.ensure(lambda result: result is not None, "Audio Onset Detection output must not be None")
def audio_onset_detection(signal: NDArray[np.float64], sampling_rate: float, sm_size: int, size: int, alpha: float, wrange: int | tuple[int, int] | range, d1_th: float, d2_th: float) -> NDArray[np.int_]:
    """Detects note/event onset locations from an input audio signal using the Zong (2003) procedure and threshold/window parameters.

    Args:
        signal: Audio waveform samples.
        sampling_rate: Must be positive.
        sm_size: Smoothing size; typically > 0.
        size: Analysis window/feature size; typically > 0.
        alpha: Algorithm weighting/sensitivity parameter.
        wrange: Search/window range for onset decision.
        d1_th: First-derivative threshold.
        d2_th: Second-derivative threshold.

    Returns:
        Detected onset positions (e.g., sample indices or times).
    """
    return find_onsets_zong2003(signal=signal, sampling_rate=sampling_rate, sm_size=sm_size, size=size, alpha=alpha, wrange=wrange, d1_th=d1_th, d2_th=d2_th)["onsets"]
