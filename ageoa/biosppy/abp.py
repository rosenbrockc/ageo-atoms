from __future__ import annotations

import numpy as np; from numpy.typing import NDArray

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from .abp_witnesses import witness_audio_onset_detection
from biosppy.signals.abp import find_onsets_zong2003

@register_atom(witness_audio_onset_detection)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda sampling_rate: sampling_rate is not None, "sampling_rate cannot be None")
@icontract.ensure(lambda result: result is not None, "Audio Onset Detection output must not be None")
def audio_onset_detection(
    signal: NDArray[np.float64],
    sampling_rate: float = 1000.0,
    sm_size: int | None = None,
    size: int | None = None,
    alpha: float = 2.0,
    wrange: int | tuple[int, int] | range | None = None,
    d1_th: float = 0.0,
    d2_th: float | None = None,
) -> NDArray[np.int_]:
    """Detects note/event onset locations from an input audio signal using the Zong (2003) procedure and threshold/window parameters.

    Args:
        signal: Audio waveform samples.
        sampling_rate: Must be positive.
        sm_size: Optional smoothing size.
        size: Optional analysis window size.
        alpha: Algorithm weighting/sensitivity parameter.
        wrange: Optional search/window range for onset decision.
        d1_th: First-derivative threshold.
        d2_th: Optional second-derivative threshold.

    Returns:
        Detected onset positions (e.g., sample indices or times).
    """
    return find_onsets_zong2003(signal=signal, sampling_rate=sampling_rate, sm_size=sm_size, size=size, alpha=alpha, wrange=wrange, d1_th=d1_th, d2_th=d2_th)["onsets"]
