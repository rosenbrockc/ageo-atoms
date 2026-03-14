from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_tt2tdb_offset(seconds: AbstractArray) -> AbstractArray:
    """Shape-and-type check for tt2 tdb offset. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=seconds.shape,
        dtype="float64",
    )
    return result