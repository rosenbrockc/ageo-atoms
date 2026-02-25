"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

# Witness functions should be imported from the generated witnesses module
def witness_detectonsetevents(*args: object, **kwargs: object) -> bool:
    return True

@register_atom(witness_detectonsetevents)  # type: ignore[untyped-decorator]
@icontract.require(lambda signal: isinstance(signal, (float, int, np.number)), "signal must be numeric")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda alpha: isinstance(alpha, (float, int, np.number)), "alpha must be numeric")
@icontract.require(lambda init_bpm: isinstance(init_bpm, (float, int, np.number)), "init_bpm must be numeric")
@icontract.require(lambda min_delay: isinstance(min_delay, (float, int, np.number)), "min_delay must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "DetectOnsetEvents output must not be None")
def detectonsetevents(signal: np.ndarray[tuple[int, ...], np.dtype[np.float64]], sampling_rate: float, alpha: float, k: int, init_bpm: float, min_delay: float, max_BPM: float) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Detects rhythmic onset events from an input signal using provided tempo and delay constraints as algorithm parameters.

    Args:
        signal: 1-D sampled signal
        sampling_rate: > 0
        alpha: algorithm coefficient
        k: window/order parameter, typically > 0
        init_bpm: initial tempo estimate, > 0
        min_delay: minimum inter-onset delay, >= 0
        max_BPM: upper tempo bound, > 0

    Returns:
        detected onset locations/times; may be empty
    """
    raise NotImplementedError("Wire to original implementation")