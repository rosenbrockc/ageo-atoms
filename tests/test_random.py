"""Tests for ageoa.numpy.random atoms."""

import numpy as np
import pytest
import icontract
from ageoa.numpy import random as ag_random

class TestRand:
    """Tests for the rand atom."""

    def test_positive_basic(self):
        r = ag_random.rand((2, 3))
        assert r.shape == (2, 3)
        assert np.all(r >= 0) and np.all(r < 1)

    def test_require_size_type(self):
        with pytest.raises(icontract.ViolationError, match="Size must be"):
            ag_random.rand("invalid")

    def test_postcondition_shape(self):
        shape = (4, 5)
        r = ag_random.rand(shape)
        assert r.shape == shape

    def test_scalar_result(self):
        r = ag_random.rand(None)
        assert isinstance(r, float)
        assert 0 <= r < 1

    def test_matches_upstream(self):
        # np.random.rand doesn't take size as a tuple like our atom, 
        # so we check behavior on single int
        np.random.seed(42)
        r_raw = np.random.rand(5)
        np.random.seed(42)
        # Assuming our atom uses rand(*size) internally
        # Let's re-seed both
        ag_seed = ag_random.default_rng(42) # Wait, rand uses global seed or Generator?
        # Rand uses legacy global state.
        np.random.seed(42)
        r_atom = ag_random.rand(5)
        np.random.seed(42)
        r_raw = np.random.rand(5)
        np.testing.assert_array_equal(r_atom, r_raw)

class TestUniform:
    """Tests for the uniform atom."""

    def test_positive_basic(self):
        u = ag_random.uniform(low=10, high=20, size=(5,))
        assert u.shape == (5,)
        assert np.all(u >= 10) and np.all(u < 20)

    def test_require_low_high(self):
        with pytest.raises(icontract.ViolationError, match="low must be"):
            ag_random.uniform(low=20, high=10)

    def test_postcondition_shape(self):
        shape = (3, 2)
        u = ag_random.uniform(size=shape)
        assert u.shape == shape

    def test_default_range(self):
        u = ag_random.uniform(size=100)
        assert np.all(u >= 0) and np.all(u < 1)

    def test_matches_upstream(self):
        np.random.seed(42)
        u_atom = ag_random.uniform(0, 10, size=5)
        np.random.seed(42)
        u_raw = np.random.uniform(0, 10, size=5)
        np.testing.assert_array_equal(u_atom, u_raw)

class TestDefaultRng:
    """Tests for the default_rng atom."""

    def test_positive_basic(self):
        rng = ag_random.default_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_require_seed_type(self):
        with pytest.raises(icontract.ViolationError, match="Invalid seed type"):
            ag_random.default_rng("invalid")

    def test_postcondition_type(self):
        rng = ag_random.default_rng()
        assert isinstance(rng, np.random.Generator)

    def test_fixed_seed_consistency(self):
        rng1 = ag_random.default_rng(42)
        rng2 = ag_random.default_rng(42)
        assert rng1.random() == rng2.random()

    def test_matches_upstream(self):
        rng_atom = ag_random.default_rng(123)
        rng_raw = np.random.default_rng(123)
        assert rng_atom.random() == rng_raw.random()
