from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_encodedistancematrix(mat_list: AbstractArray, max_cdr3: AbstractScalar, max_epi: AbstractScalar) -> AbstractArray:
    """Ghost witness for EncodeDistanceMatrix."""
    result = AbstractArray(
        shape=mat_list.shape,
        dtype="float64",
    )
    return result
