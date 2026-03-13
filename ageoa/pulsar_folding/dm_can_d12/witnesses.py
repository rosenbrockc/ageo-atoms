from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_dm_candidate_filter(data: AbstractArray, data_base: AbstractArray, sens: AbstractArray, DM_base: AbstractArray, candidates: AbstractArray, fchan: AbstractArray, width: AbstractArray, tsamp: AbstractArray) -> AbstractArray:
    """Ghost witness for DM_candidate_filter."""
    result = AbstractArray(
        shape=data.shape,
        dtype="float64",
    )
    return result
