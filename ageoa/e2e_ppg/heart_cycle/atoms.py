"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_detect_heart_cycles)
@icontract.require(lambda ppg: isinstance(ppg, np.ndarray), "ppg must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "detect_heart_cycles output must not be None")
def detect_heart_cycles(ppg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Detects heart cycles from a PPG signal.

    Args:
        ppg: Raw PPG signal.
        sampling_rate: The sampling frequency of the PPG signal.

    Returns:
        Indices of detected heart cycles.
    """
    raise NotImplementedError("Wire to original implementation")
