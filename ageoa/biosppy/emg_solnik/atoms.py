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

def witness_threshold_based_onset_detection(*args: object, **kwargs: object) -> object:
    return None
@register_atom(witness_threshold_based_onset_detection)
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda threshold: isinstance(threshold, (float, int, np.number)), "threshold must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "Threshold-Based Onset Detection output must not be None")
def threshold_based_onset_detection(signal: object, rest: object, sampling_rate: float, threshold: float, active_state_duration: float) -> object:  # type: ignore[untyped-decorator]
    """Detects activation onset points in a signal by comparing against a rest-derived baseline and enforcing a minimum active-state duration.

    Args:
        signal: 1D time-series samples
        rest: baseline/rest segment used to calibrate detection
        sampling_rate: > 0, samples per second
        threshold: activation cutoff relative to baseline/rest statistics
        active_state_duration: minimum sustained active duration (seconds or equivalent sample window)

    Returns:
        indices marking detected onsets that satisfy threshold and duration criteria
    """
    raise NotImplementedError("Wire to original implementation")