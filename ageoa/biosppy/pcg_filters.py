from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .pcg_filters_witnesses import witness_homomorphic_signal_filtering
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

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .pcg_filters_witnesses import witness_homomorphicfilter
from biosppy.signals.pcg import homomorphic_filter

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_homomorphicfilter)  # type: ignore[name-defined, untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "HomomorphicFilter output must not be None")
def homomorphicfilter(signal: np.ndarray, sampling_rate: float) -> np.ndarray:  # type: ignore[type-arg]
    """Applies a homomorphic filter to a phonocardiogram (PCG) heart-sound signal. Takes the logarithm, filters in the frequency domain, then exponentiates back to extract the signal envelope with compressed dynamic range.

    Args:
        signal: non-zero values required before log transform; length >= 1
        sampling_rate: must be > 0

    Returns:
        same shape as input signal
    """
    return homomorphic_filter(signal=signal, sampling_rate=sampling_rate)["homomorphic_envelope"]
