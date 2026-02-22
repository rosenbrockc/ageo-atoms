"""Tests for mint."""

import pytest
import numpy as np
import icontract
from ageoa.mint.atoms import axial_attention, rotary_positional_embeddings


class TestAxialAttention:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            axial_attention(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            axial_attention(None)

    def test_precondition_wrong_ndim(self):
        with pytest.raises(icontract.ViolationError):
            axial_attention(np.array([1.0, 2.0]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            axial_attention(np.array([[np.nan, 1.0], [2.0, 3.0]]))


class TestRotaryPositionalEmbeddings:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            rotary_positional_embeddings(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            rotary_positional_embeddings(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            rotary_positional_embeddings(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            rotary_positional_embeddings(np.array([np.inf]))
