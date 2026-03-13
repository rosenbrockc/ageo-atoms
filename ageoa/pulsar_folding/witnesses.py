from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_dm_can_brute_force(data: AbstractArray) -> AbstractArray:
    """Ghost witness for dm_can_brute_force."""
    result = AbstractArray(
        shape=data.shape,
        dtype="float64",
    )
    return result


def witness_spline_bandpass_correction(data: AbstractArray) -> AbstractArray:
    """Ghost witness for spline_bandpass_correction."""
    result = AbstractArray(
        shape=data.shape,
        dtype="float64",
    )
    return result
