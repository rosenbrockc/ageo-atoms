from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_foot_sensing_state_update(foot_sensing_state_in, *args, **kwargs):
    """Shape-and-type check for foot sensing state update. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=foot_sensing_state_in.shape,
        dtype="float64",)

    return result

def witness_mode_snapshot_readout(mode_state_in: AbstractArray) -> AbstractArray:
    """Shape-and-type check for mode snapshot readout. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=mode_state_in.shape,
        dtype="float64",)

    return result
