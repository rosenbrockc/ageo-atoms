"""Tests for sorting algorithm atoms (Issue 11)."""

import numpy as np
import pytest

from ageoa.algorithms.sorting import (
    counting_sort,
    heapsort,
    merge_sort,
    quicksort,
    radix_sort,
)
from ageoa.ghost.abstract import AbstractArray
from ageoa.ghost.registry import REGISTRY
from ageoa.ghost.witnesses import witness_sort


class TestWitnessSort:
    def test_preserves_shape(self):
        arr = AbstractArray(shape=(100,), dtype="float64")
        result = witness_sort(arr)
        assert result.shape == (100,)
        assert result.dtype == "float64"

    def test_marks_sorted(self):
        arr = AbstractArray(shape=(50,), dtype="int64")
        result = witness_sort(arr)
        assert result.is_sorted is True

    def test_preserves_value_range(self):
        arr = AbstractArray(shape=(10,), dtype="float64", min_val=0.0, max_val=100.0)
        result = witness_sort(arr)
        assert result.min_val == 0.0
        assert result.max_val == 100.0


class TestMergeSort:
    def test_sorts_array(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = merge_sort(a)
        expected = np.array([1, 1, 2, 3, 4, 5, 6, 9])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_length(self):
        a = np.array([5, 3, 1])
        assert len(merge_sort(a)) == 3

    def test_registered(self):
        assert "merge_sort" in REGISTRY


class TestQuicksort:
    def test_sorts_array(self):
        a = np.array([10, 7, 3, 1])
        result = quicksort(a)
        np.testing.assert_array_equal(result, np.array([1, 3, 7, 10]))

    def test_registered(self):
        assert "quicksort" in REGISTRY


class TestHeapsort:
    def test_sorts_array(self):
        a = np.array([9, 1, 5, 3])
        result = heapsort(a)
        np.testing.assert_array_equal(result, np.array([1, 3, 5, 9]))

    def test_registered(self):
        assert "heapsort" in REGISTRY


class TestCountingSort:
    def test_sorts_integers(self):
        a = np.array([4, 2, 2, 8, 3, 3, 1], dtype=np.int64)
        result = counting_sort(a)
        np.testing.assert_array_equal(result, np.array([1, 2, 2, 3, 3, 4, 8]))

    def test_preserves_length(self):
        a = np.array([3, 1, 2], dtype=np.int64)
        assert len(counting_sort(a)) == 3

    def test_registered(self):
        assert "counting_sort" in REGISTRY


class TestRadixSort:
    def test_sorts_non_negative(self):
        a = np.array([170, 45, 75, 90, 802, 24, 2, 66], dtype=np.int64)
        result = radix_sort(a)
        np.testing.assert_array_equal(
            result, np.array([2, 24, 45, 66, 75, 90, 170, 802])
        )

    def test_registered(self):
        assert "radix_sort" in REGISTRY
