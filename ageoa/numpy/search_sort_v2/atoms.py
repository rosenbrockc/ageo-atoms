"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_binarysearchinsertion, witness_lexicographicindirectsort, witness_partialsortpartition

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_binarysearchinsertion)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda v: v is not None, "v cannot be None")
@icontract.require(lambda side: side is not None, "side cannot be None")
@icontract.ensure(lambda result: result is not None, "BinarySearchInsertion output must not be None")
def binarysearchinsertion(a: np.ndarray, v: np.ndarray, side: str = 'left', sorter: Optional[np.ndarray] = None) -> np.ndarray:
    """Locates insertion points for values into a sorted array using binary search, supporting left/right side preference and an optional indirect sorter index.

    Args:
        a: must be sorted along search axis unless sorter is provided
        v: broadcastable against a
        side: determines which boundary index is returned on a tie
        sorter: indices that sort a; if None, a must already be sorted

    Returns:
        values in [0, len(a)]; same shape as v
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_lexicographicindirectsort)
@icontract.require(lambda keys: keys is not None, "keys cannot be None")
@icontract.ensure(lambda result: result is not None, "LexicographicIndirectSort output must not be None")
def lexicographicindirectsort(keys: Sequence[np.ndarray], axis: int = -1) -> np.ndarray:
    """Performs lexicographic indirect sort of a sequence of keys.

    Args:
        keys: all arrays must have the same shape
        axis: axis along which to sort; default -1

    Returns:
        permutation that sorts keys lexicographically; shape matches key array shape along sort axis
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_partialsortpartition)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda kth: kth is not None, "kth cannot be None")
@icontract.require(lambda axis: axis is not None, "axis cannot be None")
@icontract.require(lambda kind: kind is not None, "kind cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "PartialSortPartition all outputs must not be None")
def partialsortpartition(a: np.ndarray, kth: Union[int, Sequence[int]], axis: Optional[int] = -1, kind: str = 'introselect', order: Optional[Union[str, List[str]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Performs partial sort partitioning of an array.

    Args:
        a: input array to partition
        kth: index/indices of elements to place in sorted position
        axis: axis along which to partition; None flattens first
        kind: selection algorithm; currently only introselect supported
        order: field ordering for structured arrays; ignored for plain ndarrays

    Returns:
        partitioned_array: same shape/dtype as a; k-th elements in correct sorted positions (from partition)
        partition_indices: indirect permutation achieving the partition; same shape as a (from argpartition)
    """
    raise NotImplementedError("Wire to original implementation")
