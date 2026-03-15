from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_detect_heart_cycles
from ppg_sqa import heart_cycle_detection

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_detect_heart_cycles)
@icontract.require(lambda ppg: isinstance(ppg, np.ndarray), "ppg must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "detect_heart_cycles output must not be None")
def detect_heart_cycles(ppg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Detects individual heart cycles from a photoplethysmography (PPG) signal — an optical measurement of blood volume changes, typically recorded from a fingertip or wrist sensor.

    Args:
        ppg: Raw PPG signal.
        sampling_rate: The sampling frequency of the PPG signal.

    Returns:
        Indices of detected heart cycle boundaries.
    """
    return heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate)
