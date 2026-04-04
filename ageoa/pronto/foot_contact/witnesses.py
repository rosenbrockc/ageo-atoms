from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray

def witness_foot_sensing_state_update(foot_sensing_state_in: AbstractArray, *args, **kwargs) -> AbstractArray:
    """Shape-and-type check for foot sensing state update. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=foot_sensing_state_in.shape,
        dtype="float64",)

    return result

def witness_mode_snapshot_readout(mode_state_in: AbstractArray) -> AbstractArray:
    """Describe the two-value mode snapshot returned from immutable state."""
    return AbstractArray(shape=("2",), dtype="tuple[hashable,hashable]")
