"""Tests for mint."""

import pytest
import numpy as np
import icontract
from ageoa.mint.atoms import axial_attention, rotary_positional_embeddings


class TestAxialAttention:
    def test_returns_result(self):
        x = np.arange(16, dtype=np.float64).reshape(2, 2, 1, 4)
        result = axial_attention(x)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            axial_attention(None)

    def test_precondition_wrong_ndim(self):
        with pytest.raises(icontract.ViolationError):
            axial_attention(np.array([1.0, 2.0]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            axial_attention(np.array([[[[np.nan, 1.0], [2.0, 3.0]]]]))


class TestRotaryPositionalEmbeddings:
    def test_returns_result(self):
        q = np.arange(8, dtype=np.float64).reshape(1, 2, 4)
        k = q + 1.0
        result = rotary_positional_embeddings(q, k)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            rotary_positional_embeddings(None, np.zeros((1, 2, 4)))

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            rotary_positional_embeddings(np.array([]), np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            rotary_positional_embeddings(np.array([[[np.inf, 0.0]]]), np.array([[[0.0, 1.0]]]))
