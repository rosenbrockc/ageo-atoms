from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_binarysearchinsertion(a: AbstractArray, v: AbstractArray, side: AbstractArray, sorter: AbstractArray) -> AbstractArray:
    """Shape-and-type check for binary search insertion. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result

def witness_lexicographicindirectsort(keys: AbstractArray, axis: AbstractArray) -> AbstractArray:
    """Shape-and-type check for lexicographic indirect sort. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=keys.shape,
        dtype="float64",
    )
    return result

def witness_partialsortpartition(a: AbstractArray, kth: AbstractArray, axis: AbstractArray, kind: AbstractArray, order: AbstractArray) -> AbstractArray:
    """Shape-and-type check for partial sort partition. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result
