"""Search algorithm atoms with icontract contracts and ghost witnesses.

Covers binary search, linear search, and hash-based lookup.
"""

from __future__ import annotations

from typing import Any

import icontract
import numpy as np

from ageoa.ghost.abstract import AbstractArray, AbstractScalar
from ageoa.ghost.registry import register_atom
from ageoa.ghost.witnesses import witness_search


def _witness_binary_search(arr: AbstractArray, key: AbstractScalar) -> AbstractScalar:
    """Ghost witness for binary search: requires sorted input."""
    if not arr.is_sorted:
        raise ValueError("Binary search requires a sorted array")
    return witness_search(arr, key)


@register_atom(witness=_witness_binary_search)
@icontract.require(lambda arr: len(arr) > 0, "Array must be non-empty")
@icontract.require(
    lambda arr: all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)),
    "Array must be sorted for binary search",
)
@icontract.ensure(
    lambda result: result >= -1,
    "Result must be valid index or -1",
)
def binary_search(arr: np.ndarray, key: Any) -> int:
    """Binary search: O(log n) search in sorted array.

    Returns index of key if found, -1 otherwise.

    Args:
        arr: Non-empty sorted 1D input array.
        key: Value to search for.

    Returns:
        Index of key if found, -1 otherwise.
    """
    idx = np.searchsorted(arr, key)
    if idx < len(arr) and arr[idx] == key:
        return int(idx)
    return -1


def _witness_linear_search(arr: AbstractArray, key: AbstractScalar) -> AbstractScalar:
    """Ghost witness for linear search: no preconditions on ordering."""
    return witness_search(arr, key)


@register_atom(witness=_witness_linear_search)
@icontract.require(lambda arr: len(arr) > 0, "Array must be non-empty")
@icontract.ensure(
    lambda result: result >= -1,
    "Result must be valid index or -1",
)
def linear_search(arr: np.ndarray, key: Any) -> int:
    """Linear search: O(n) search in unsorted array.

    Returns index of first occurrence of key, -1 if not found.

    Args:
        arr: Non-empty 1D input array.
        key: Value to search for.

    Returns:
        Index of first occurrence of key, or -1 if not found.
    """
    matches = np.where(arr == key)[0]
    if len(matches) > 0:
        return int(matches[0])
    return -1


def _witness_hash_lookup(arr: AbstractArray, key: AbstractScalar) -> AbstractScalar:
    """Ghost witness for hash lookup: O(1) amortized."""
    return AbstractScalar(
        dtype="int64",
        min_val=-1,
        max_val=float(arr.shape[0] - 1) if arr.shape else 0,
        is_index=True,
    )


@register_atom(witness=_witness_hash_lookup)
@icontract.require(lambda arr: len(arr) > 0, "Array must be non-empty")
@icontract.ensure(
    lambda result: result >= -1,
    "Result must be valid index or -1",
)
def hash_lookup(arr: np.ndarray, key: Any) -> int:
    """Hash-based lookup: O(1) amortized via dict index.

    Builds a hash table from array values to indices. Returns the
    index of the first occurrence of key, or -1 if not found.

    Args:
        arr: Non-empty 1D input array.
        key: Value to search for.

    Returns:
        Index of first occurrence of key, or -1 if not found.
    """
    table = {}
    for i, v in enumerate(arr):
        val = v.item() if hasattr(v, "item") else v
        if val not in table:
            table[val] = i
    k = key.item() if hasattr(key, "item") else key
    return table.get(k, -1)
