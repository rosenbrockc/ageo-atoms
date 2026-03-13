from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_tt2tdb_offset(seconds: AbstractArray) -> AbstractArray:
    """Ghost witness for TT2TDB_Offset."""
    result = AbstractArray(
        shape=seconds.shape,
        dtype="float64",
    )
    return result