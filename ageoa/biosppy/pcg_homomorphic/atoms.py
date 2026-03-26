from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import witness_homomorphic_signal_filtering
from biosppy.signals.pcg import homomorphic_filter

@register_atom(witness_homomorphic_signal_filtering)
@icontract.require(lambda signal: signal.ndim >= 1, "signal must be at least 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be np.ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def homomorphic_signal_filtering(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Apply homomorphic filtering to an input signal using the provided sampling rate.

    Args:
        signal: Input signal; 1-D or compatible tensor.
        sampling_rate: Positive sampling frequency in Hz.

    Returns:
        Filtered output signal with same temporal support as input.
    """
    return homomorphic_filter(signal=signal, sampling_rate=sampling_rate)["homomorphic_envelope"]
