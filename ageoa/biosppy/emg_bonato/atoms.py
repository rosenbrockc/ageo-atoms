"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_bonato_onset_detection)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.require(lambda rest: isinstance(rest, np.ndarray), "rest must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda threshold: isinstance(threshold, (float, int, np.number)), "threshold must be numeric")
@icontract.require(lambda active_state_duration: isinstance(active_state_duration, (float, int, np.number)), "active_state_duration must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "bonato_onset_detection output must not be None")
def bonato_onset_detection(signal: np.ndarray, rest: np.ndarray, sampling_rate: float, threshold: float, active_state_duration: float, samples_above_fail: int, fail_size: int) -> List[int]:
    """Detects activity onsets in a signal using the Bonato double-threshold algorithm. It identifies points where the signal exceeds a defined threshold for a minimum duration.

    Args:
        signal: Input signal trace as a 1D numpy array.
        rest: Signal segment corresponding to a rest period, used for noise estimation.
        sampling_rate: The signal's sampling frequency in Hz.
        threshold: The amplitude threshold for marking a potential onset.
        active_state_duration: The minimum duration (in seconds) an active state must be maintained to be confirmed.
        samples_above_fail: Number of consecutive samples that must be above the threshold to initiate an active state.
        fail_size: Size of the window (in samples) to check for validation after a potential onset.

    Returns:
        A list of integer indices corresponding to the detected onset locations in the signal.
    """
    raise NotImplementedError("Wire to original implementation")
