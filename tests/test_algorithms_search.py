"""Tests for search algorithm atoms (Issue 11)."""

import numpy as np
import pytest

from ageoa.algorithms.search import binary_search, hash_lookup, linear_search
from ageoa.ghost.registry import REGISTRY


class TestBinarySearch:
    def test_finds_element(self):
        arr = np.array([1, 3, 5, 7, 9])
        assert binary_search(arr, 5) == 2

    def test_not_found(self):
        arr = np.array([1, 3, 5, 7, 9])
        assert binary_search(arr, 4) == -1

    def test_first_element(self):
        arr = np.array([1, 3, 5])
        assert binary_search(arr, 1) == 0

    def test_last_element(self):
        arr = np.array([1, 3, 5])
        assert binary_search(arr, 5) == 2

    def test_registered(self):
        assert "binary_search" in REGISTRY


class TestLinearSearch:
    def test_finds_element(self):
        arr = np.array([5, 3, 1, 7, 9])
        assert linear_search(arr, 7) == 3

    def test_not_found(self):
        arr = np.array([5, 3, 1])
        assert linear_search(arr, 42) == -1

    def test_first_occurrence(self):
        arr = np.array([1, 2, 1, 3])
        assert linear_search(arr, 1) == 0

    def test_registered(self):
        assert "linear_search" in REGISTRY


class TestHashLookup:
    def test_finds_element(self):
        arr = np.array([10, 20, 30, 40])
        assert hash_lookup(arr, 30) == 2

    def test_not_found(self):
        arr = np.array([10, 20, 30])
        assert hash_lookup(arr, 99) == -1

    def test_registered(self):
        assert "hash_lookup" in REGISTRY
