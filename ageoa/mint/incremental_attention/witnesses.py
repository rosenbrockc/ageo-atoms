from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_enable_incremental_state_configuration(cls: AbstractArray) -> AbstractArray:
    """Ghost witness for enable_incremental_state_configuration."""
    result = AbstractArray(
        shape=cls.shape,
        dtype="float64",
    )
    return result
