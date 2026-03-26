from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_homomorphicfilter
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
