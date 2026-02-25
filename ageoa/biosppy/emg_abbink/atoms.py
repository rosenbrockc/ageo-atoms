"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from ageoa.ghost.registry import register_atom

# Witness functions should be imported from the generated witnesses module
@register_atom(witness_detect_onsets_with_rest_aware_thresholds)  # type: ignore[name-defined,untyped-decorator]
@register_atom(witness_detect_onsets_with_rest_aware_thresholds)
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda threshold: isinstance(threshold, (float, int, np.number)), "threshold must be numeric")
@icontract.require(lambda transition_threshold: isinstance(transition_threshold, (float, int, np.number)), "transition_threshold must be numeric")
def detect_onsets_with_rest_aware_thresholds(signal: npt.NDArray[np.float64], rest: npt.NDArray[np.float64] | float | int, sampling_rate: float, size: int, alarm_size: int, threshold: float, transition_threshold: float) -> npt.NDArray[np.float64]:
    """Detects onset events from an input signal using rest/reference information, window sizes, sampling rate, and transition/alarm thresholds.

    Args:
        signal: non-empty; length should support window operations
        rest: compatible with signal domain
        sampling_rate: > 0
        size: > 0
        alarm_size: > 0
        threshold: application-defined detection threshold
        transition_threshold: application-defined transition threshold

    Returns:
        aligned with input signal timeline
    """
    raise NotImplementedError("Wire to original implementation")