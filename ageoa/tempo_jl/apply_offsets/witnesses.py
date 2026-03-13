from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_show(io: AbstractScalar, s: AbstractScalar) -> AbstractScalar:
    """Ghost witness for Show."""
    result = AbstractScalar(
        dtype="str",
    )
    return result


def witness__zero_offset(seconds: AbstractScalar) -> AbstractScalar:
    """Ghost witness for Zero Offset."""
    result = AbstractScalar(
        dtype="float64",
    )
    return result


def witness_apply_offsets(sec: AbstractScalar, ts1: AbstractScalar, ts2: AbstractScalar) -> AbstractScalar:
    """Ghost witness for Apply Offsets."""
    result = AbstractScalar(
        dtype="float64",
    )
    return result
