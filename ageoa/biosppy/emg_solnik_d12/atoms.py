from __future__ import annotations
import icontract
"""Auto-generated atom wrappers following the ageoa pattern."""


from typing import Any
import numpy as np

from ageoa.ghost.registry import register_atom
from .witnesses import witness_solnik_onset_detect
@register_atom(witness_solnik_onset_detect)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda rest: rest is not None, "rest cannot be None")
@icontract.require(lambda sampling_rate: sampling_rate is not None, "sampling_rate cannot be None")
@icontract.require(lambda threshold: threshold is not None, "threshold cannot be None")
@icontract.require(lambda active_state_duration: active_state_duration is not None, "active_state_duration cannot be None")
@icontract.ensure(lambda result: result is not None, "solnik_onset_detect output must not be None")
def solnik_onset_detect(signal: np.ndarray, rest: float, sampling_rate: float, threshold: float, active_state_duration: float) -> np.ndarray:
    """Detects movement onsets in a signal using the Solnik algorithm: identifies transitions from rest to active state by comparing signal amplitude against a threshold over a minimum active-state duration window.

    Args:
        signal: non-empty, finite values
        rest: finite, typically >= 0
        sampling_rate: > 0
        threshold: > 0
        active_state_duration: > 0

    Returns:
        values in [0, len(signal)-1], monotonically increasing
    """
    raise NotImplementedError("Wire to original implementation")
