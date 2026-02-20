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
    """Merge sort: O(n log n) stable comparison sort.

    <!-- conceptual_profile
    {
        "abstract_name": "Stable Recursive Sequence Ordering Transformer",
        "conceptual_transform": "Reorganizes an N-dimensional sequence into a monotonically non-decreasing order using a stable divide-and-conquer approach. It preserves the relative order of equal elements while ensuring global monotonic constraints.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D tensor of comparable elements (e.g., scalars or structured objects with a defined total ordering)."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of the same elements, reorganized to satisfy monotonic non-decreasing constraints."
            }
        ],
        "algorithmic_properties": [
            "divide-and-conquer",
            "stable",
            "deterministic",
            "comparison-based"
        ],
        "cross_disciplinary_applications": [
            "Ordering chronological events in a distributed system trace while preserving local causality.",
            "Ordering sequential sensor readings by timestamp for temporal alignment.",
            "Organizing experimental data points for interpolation and curve fitting."
        ]
    }
    /conceptual_profile -->
    """
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
    """Quicksort: O(n log n) average-case comparison sort.

    <!-- conceptual_profile
    {
        "abstract_name": "Partitioning-Based Sequence Ordering Transformer",
        "conceptual_transform": "Reorganizes a sequence into a monotonically non-decreasing order by iteratively partitioning elements around a pivot. It achieves efficient global ordering through localized comparative shifts.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D tensor of comparable elements."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of the same elements, reorganized to satisfy monotonic non-decreasing constraints."
            }
        ],
        "algorithmic_properties": [
            "partitioning",
            "unstable",
            "average-case-optimal",
            "comparison-based"
        ],
        "cross_disciplinary_applications": [
            "Rapidly indexing large-scale genomic sequences for fast retrieval.",
            "Optimizing search queries in high-frequency trading databases.",
            "Preprocessing point cloud data for spatial indexing and neighbor search."
        ]
    }
    /conceptual_profile -->
    """
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
    """Heapsort: O(n log n) in-place comparison sort.

    <!-- conceptual_profile
    {
        "abstract_name": "Priority-Structure Selection Reorderer",
        "conceptual_transform": "Orders a sequence by maintaining a partial ordering structure (heap) and repeatedly extracting the maximum/minimum element. It provides guaranteed performance by leveraging tree-structured data invariants.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D tensor of comparable elements."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of the same elements, reorganized to satisfy monotonic non-decreasing constraints."
            }
        ],
        "algorithmic_properties": [
            "selection-based",
            "unstable",
            "tree-structured",
            "guaranteed-worst-case"
        ],
        "cross_disciplinary_applications": [
            "Scheduling priority tasks in real-time operating systems.",
            "Implementing memory-constrained sorting in embedded hardware controllers.",
            "Extracting top-K features from a large dataset for dimensionality reduction."
        ]
    }
    /conceptual_profile -->
    """
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
    """Counting sort: O(n + k) non-comparison sort for integer arrays.

    <!-- conceptual_profile
    {
        "abstract_name": "Frequency-Distribution Discrete Value Reconstructor",
        "conceptual_transform": "Reorders a sequence of discrete, bounded values by calculating a histogram of their occurrences and reconstructing the sequence based on accumulated counts. It bypasses comparisons in favor of direct positional calculation.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D tensor of discrete integer values within a known, finite range."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of the same elements, reorganized into non-decreasing order."
            }
        ],
        "algorithmic_properties": [
            "non-comparison",
            "stable",
            "histogram-based",
            "linear-time-capable"
        ],
        "cross_disciplinary_applications": [
            "Normalizing pixel intensity histograms in digital image processing.",
            "Sorting network packets by priority level in a high-speed router.",
            "Aggregating categorical survey results for statistical analysis."
        ]
    }
    /conceptual_profile -->
    """
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
    """Radix sort: O(d * n) non-comparison sort for non-negative integers.

    <!-- conceptual_profile
    {
        "abstract_name": "Multi-Key Positional Sequence Reorderer",
        "conceptual_transform": "Orders a sequence of multi-component or multi-digit keys by performing successive stable reorderings on each component/digit. It leverages the structure of the data representation to achieve efficient ordering without comparisons.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D tensor of non-negative integers or multi-component keys."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of the same elements, reorganized into non-decreasing order."
            }
        ],
        "algorithmic_properties": [
            "non-comparison",
            "stable",
            "iterative-positional",
            "linear-time-complexity"
        ],
        "cross_disciplinary_applications": [
            "Ordering variable-length character strings in large-scale lexicographical indexing.",
            "Sorting fixed-point telemetry data in telemetry processing pipelines.",
            "Organizing IP addresses for efficient routing table lookups."
        ]
    }
    /conceptual_profile -->
    """
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
