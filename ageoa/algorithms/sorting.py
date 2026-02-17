"""Sorting algorithm atoms with icontract contracts and ghost witnesses.

Covers the 5 CLRS sorting primitives: merge_sort, quicksort, heapsort,
counting_sort, radix_sort.
"""

from __future__ import annotations

from typing import Any

import icontract
import numpy as np

from ageoa.ghost.abstract import AbstractArray
from ageoa.ghost.registry import register_atom
from ageoa.ghost.witnesses import witness_sort


@register_atom(witness=witness_sort)
@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.ensure(
    lambda result, a: len(result) == len(a),
    "Output must have same length as input",
)
@icontract.ensure(
    lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)),
    "Output must be sorted",
)
def merge_sort(a: np.ndarray) -> np.ndarray:
    """Merge sort: O(n log n) stable comparison sort."""
    return np.sort(a, kind="mergesort")


@register_atom(witness=witness_sort)
@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.ensure(
    lambda result, a: len(result) == len(a),
    "Output must have same length as input",
)
@icontract.ensure(
    lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)),
    "Output must be sorted",
)
def quicksort(a: np.ndarray) -> np.ndarray:
    """Quicksort: O(n log n) average-case comparison sort."""
    return np.sort(a, kind="quicksort")


@register_atom(witness=witness_sort)
@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.ensure(
    lambda result, a: len(result) == len(a),
    "Output must have same length as input",
)
@icontract.ensure(
    lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)),
    "Output must be sorted",
)
def heapsort(a: np.ndarray) -> np.ndarray:
    """Heapsort: O(n log n) in-place comparison sort."""
    return np.sort(a, kind="heapsort")


def _witness_counting_sort(x: AbstractArray) -> AbstractArray:
    """Witness for counting sort: requires integer dtype."""
    if "int" not in x.dtype:
        raise ValueError(f"Counting sort requires integer dtype, got {x.dtype}")
    return AbstractArray(
        shape=x.shape,
        dtype=x.dtype,
        is_sorted=True,
        min_val=x.min_val,
        max_val=x.max_val,
    )


@register_atom(witness=_witness_counting_sort)
@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.require(lambda a: np.issubdtype(a.dtype, np.integer), "Requires integer array")
@icontract.ensure(
    lambda result, a: len(result) == len(a),
    "Output must have same length as input",
)
@icontract.ensure(
    lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)),
    "Output must be sorted",
)
def counting_sort(a: np.ndarray) -> np.ndarray:
    """Counting sort: O(n + k) non-comparison sort for integer arrays."""
    if len(a) == 0:
        return a.copy()
    min_val = int(a.min())
    max_val = int(a.max())
    count = np.zeros(max_val - min_val + 1, dtype=np.intp)
    for v in a:
        count[int(v) - min_val] += 1
    result = np.empty_like(a)
    idx = 0
    for val in range(min_val, max_val + 1):
        c = count[val - min_val]
        result[idx : idx + c] = val
        idx += c
    return result


@register_atom(witness=_witness_counting_sort)
@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.require(lambda a: np.issubdtype(a.dtype, np.integer), "Requires integer array")
@icontract.require(lambda a: np.all(a >= 0), "Requires non-negative integers")
@icontract.ensure(
    lambda result, a: len(result) == len(a),
    "Output must have same length as input",
)
@icontract.ensure(
    lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)),
    "Output must be sorted",
)
def radix_sort(a: np.ndarray) -> np.ndarray:
    """Radix sort: O(d * n) non-comparison sort for non-negative integers."""
    if len(a) == 0:
        return a.copy()
    result = a.copy()
    max_val = int(result.max())
    exp = 1
    while max_val // exp > 0:
        # Counting sort by digit
        output = np.empty_like(result)
        count = np.zeros(10, dtype=np.intp)
        for v in result:
            digit = (int(v) // exp) % 10
            count[digit] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for v in reversed(result):
            digit = (int(v) // exp) % 10
            count[digit] -= 1
            output[count[digit]] = v
        result = output
        exp *= 10
    return result
