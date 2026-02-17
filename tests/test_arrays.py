"""Tests for ageoa.numpy.arrays atoms."""

import numpy as np
import pytest
import icontract
from ageoa.numpy import arrays as ag_arrays

class TestArray:
    """Tests for the array atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        a = ag_arrays.array([1, 2, 3])
        assert isinstance(a, np.ndarray)
        assert np.all(a == [1, 2, 3])

    # -- Category 2: Precondition violations --
    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="must not be None"):
            ag_arrays.array(None)

    # -- Category 3: Postcondition verification --
    def test_postcondition_type(self):
        a = ag_arrays.array([1, 2])
        assert isinstance(a, np.ndarray)

    # -- Category 4: Edge cases --
    def test_empty_list(self):
        a = ag_arrays.array([])
        assert a.size == 0

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        data = [1, 2, 3]
        a_atom = ag_arrays.array(data, dtype=float)
        a_raw = np.array(data, dtype=float)
        np.testing.assert_array_equal(a_atom, a_raw)

class TestZeros:
    """Tests for the zeros atom."""

    def test_positive_basic(self):
        z = ag_arrays.zeros((2, 3))
        assert z.shape == (2, 3)
        assert np.all(z == 0)

    def test_require_shape_type(self):
        with pytest.raises(icontract.ViolationError, match="Shape must be"):
            ag_arrays.zeros("invalid")

    def test_postcondition_shape(self):
        shape = (4, 5)
        z = ag_arrays.zeros(shape)
        assert z.shape == shape

    def test_scalar_shape(self):
        z = ag_arrays.zeros(5)
        assert z.shape == (5,)

    def test_matches_upstream(self):
        shape = (2, 2)
        z_atom = ag_arrays.zeros(shape)
        z_raw = np.zeros(shape)
        np.testing.assert_array_equal(z_atom, z_raw)

class TestDot:
    """Tests for the dot atom."""

    def test_positive_basic(self):
        a = [1, 2]
        b = [3, 4]
        assert ag_arrays.dot(a, b) == 11

    def test_require_compatible_dims(self):
        with pytest.raises(icontract.ViolationError, match="compatible"):
            ag_arrays.dot([1, 2], [1, 2, 3])

    def test_postcondition_non_null(self):
        assert ag_arrays.dot(1, 2) is not None

    def test_matrix_vector(self):
        m = [[1, 2], [3, 4]]
        v = [5, 6]
        assert np.all(ag_arrays.dot(m, v) == [17, 39])

    def test_matches_upstream(self):
        a = np.random.rand(3, 3)
        b = np.random.rand(3, 2)
        res_atom = ag_arrays.dot(a, b)
        res_raw = np.dot(a, b)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestVstack:
    """Tests for the vstack atom."""

    def test_positive_basic(self):
        v1 = [1, 2]
        v2 = [3, 4]
        res = ag_arrays.vstack([v1, v2])
        assert np.all(res == [[1, 2], [3, 4]])

    def test_require_non_empty(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_arrays.vstack([])

    def test_postcondition_leading_dim(self):
        tup = ([1, 2], [3, 4], [5, 6])
        res = ag_arrays.vstack(tup)
        assert res.shape[0] == 3

    def test_single_array(self):
        res = ag_arrays.vstack([[1, 2]])
        assert res.shape == (1, 2)

    def test_matches_upstream(self):
        tup = (np.array([1, 2]), np.array([3, 4]))
        res_atom = ag_arrays.vstack(tup)
        res_raw = np.vstack(tup)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestReshape:
    """Tests for the reshape atom."""

    def test_positive_basic(self):
        a = np.array([1, 2, 3, 4])
        res = ag_arrays.reshape(a, (2, 2))
        assert res.shape == (2, 2)

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_arrays.reshape(None, (2, 2))

    def test_postcondition_size(self):
        a = np.zeros((3, 4))
        res = ag_arrays.reshape(a, (6, 2))
        assert res.size == a.size

    def test_minus_one_dim(self):
        a = np.zeros(10)
        res = ag_arrays.reshape(a, (2, -1))
        assert res.shape == (2, 5)

    def test_matches_upstream(self):
        a = np.random.rand(12)
        shape = (3, 4)
        res_atom = ag_arrays.reshape(a, shape)
        res_raw = np.reshape(a, shape)
        np.testing.assert_array_equal(res_atom, res_raw)
