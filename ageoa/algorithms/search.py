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

    <!-- conceptual_profile
    {
        "abstract_name": "Ordered Sequence Bisector",
        "conceptual_transform": "Locates the position of a specific target element within a monotonically ordered sequence by iteratively halving the search space.",
        "abstract_inputs": [
            {
                "name": "arr",
                "description": "A 1D tensor representing a strictly ordered sequence of elements."
            },
            {
                "name": "key",
                "description": "A scalar representing the target element to locate."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An integer representing the positional index of the target, or a negative value if not found."
            }
        ],
        "algorithmic_properties": [
            "logarithmic-time",
            "divide-and-conquer",
            "requires-ordering"
        ],
        "cross_disciplinary_applications": [
            "Finding an exact timestamp in a sorted log file.",
            "Locating threshold boundaries in sorted calibration tables.",
            "Resolving spatial coordinates in a discretized grid search."
        ]
    }
    /conceptual_profile -->
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

    <!-- conceptual_profile
    {
        "abstract_name": "Sequential Exhaustive Matcher",
        "conceptual_transform": "Scans an unordered sequence element-by-element until a condition matching the target element is satisfied.",
        "abstract_inputs": [
            {
                "name": "arr",
                "description": "A 1D tensor representing an unordered sequence of elements."
            },
            {
                "name": "key",
                "description": "A scalar representing the target element to locate."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An integer representing the positional index of the first target occurrence, or a negative value if not found."
            }
        ],
        "algorithmic_properties": [
            "linear-time",
            "exhaustive",
            "order-independent"
        ],
        "cross_disciplinary_applications": [
            "Finding a specific anomalous reading in an unsorted sensor stream.",
            "Scanning a biological sequence for a unique base pair exact match.",
            "Identifying the first failure state in a chronological test record."
        ]
    }
    /conceptual_profile -->
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

    <!-- conceptual_profile
    {
        "abstract_name": "Constant-Time Key-Value Indexer",
        "conceptual_transform": "Constructs a randomized, directly-addressable mapping of elements to their positions, allowing amortized constant-time retrieval of location given an element.",
        "abstract_inputs": [
            {
                "name": "arr",
                "description": "A 1D tensor representing an unordered sequence of elements."
            },
            {
                "name": "key",
                "description": "A scalar representing the target element to locate."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An integer representing the positional index of the target, or a negative value if not found."
            }
        ],
        "algorithmic_properties": [
            "constant-time",
            "hash-mapping",
            "memory-intensive",
            "order-independent"
        ],
        "cross_disciplinary_applications": [
            "Rapidly querying a cache of precomputed aerodynamic coefficients.",
            "Resolving MAC addresses from a highly active network switch table.",
            "Retrieving user session states in a high-throughput web application."
        ]
    }
    /conceptual_profile -->
    """
    table = {}
    for i, v in enumerate(arr):
        val = v.item() if hasattr(v, "item") else v
        if val not in table:
            table[val] = i
    k = key.item() if hasattr(key, "item") else key
    return table.get(k, -1)
