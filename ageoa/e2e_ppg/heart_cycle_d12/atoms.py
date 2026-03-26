from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_heart_cycle_detection
from ppg_sqa import heart_cycle_detection as _heart_cycle_detection


@register_atom(witness_heart_cycle_detection)
@icontract.require(lambda ppg: isinstance(ppg, np.ndarray), "ppg must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)) and sampling_rate > 0, "sampling_rate must be numeric and positive")
@icontract.ensure(lambda result: isinstance(result, list), "heart_cycle_detection must return a list")
def heart_cycle_detection(ppg: np.ndarray, sampling_rate: float) -> list[int]:
    """Detects individual heart cycles from a photoplethysmography (PPG) signal at the given sampling rate, identifying cycle boundaries or fiducial points within the waveform.

    Args:
        ppg: non-empty, finite-valued samples
        sampling_rate: must be > 0

    Returns:
        indices within [0, len(ppg))"""
    return _heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate)
