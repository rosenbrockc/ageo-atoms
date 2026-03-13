from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_foot_sensing_state_update(foot_sensing_state_in, *args, **kwargs):
    """Ghost witness for Foot Sensing State Update."""
    result = AbstractArray(
        shape=foot_sensing_state_in.shape,
        dtype="float64",)

    return result

def witness_mode_snapshot_readout(mode_state_in: AbstractArray) -> AbstractArray:
    """Ghost witness for Mode Snapshot Readout."""
    result = AbstractArray(
        shape=mode_state_in.shape,
        dtype="float64",)

    return result
