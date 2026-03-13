from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_binarysearchinsertion(a: AbstractArray, v: AbstractArray, side: AbstractArray, sorter: AbstractArray) -> AbstractArray:
    """Ghost witness for BinarySearchInsertion."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result

def witness_lexicographicindirectsort(keys: AbstractArray, axis: AbstractArray) -> AbstractArray:
    """Ghost witness for LexicographicIndirectSort."""
    result = AbstractArray(
        shape=keys.shape,
        dtype="float64",
    )
    return result

def witness_partialsortpartition(a: AbstractArray, kth: AbstractArray, axis: AbstractArray, kind: AbstractArray, order: AbstractArray) -> AbstractArray:
    """Ghost witness for PartialSortPartition."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result
