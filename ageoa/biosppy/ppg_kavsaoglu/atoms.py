"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import witness_detectonsetevents

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
        init_bpm: Initial tempo estimate in BPM; must be > 0.
        min_delay: Minimum inter-onset delay; must be >= 0.
        max_BPM: Upper tempo bound in BPM; must be > 0.

    Returns:
        Detected onset locations/times; may be empty.
    """
    raise NotImplementedError("Wire to original implementation")
