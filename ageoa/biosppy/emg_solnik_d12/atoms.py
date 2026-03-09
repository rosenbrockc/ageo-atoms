"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from ageoa.ghost.registry import register_atom
# Witness functions should be imported from the generated witnesses module
witness_solnik_onset_detect: Any = None
@register_atom(witness_solnik_onset_detect)  # type: ignore[untyped-decorator]
@register_atom(witness_solnik_onset_detect)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda rest: rest is not None, "rest cannot be None")
@icontract.require(lambda sampling_rate: sampling_rate is not None, "sampling_rate cannot be None")
@icontract.require(lambda threshold: threshold is not None, "threshold cannot be None")
@icontract.require(lambda active_state_duration: active_state_duration is not None, "active_state_duration cannot be None")
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