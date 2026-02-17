"""Tests for ageoa.numpy.fft atoms."""

import numpy as np
import pytest
import icontract
from ageoa.numpy import fft as ag_fft

class TestFft:
    """Tests for the fft atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        a = [1, 2, 3, 4]
        res = ag_fft.fft(a)
        expected = np.fft.fft(a)
        assert np.allclose(res, expected)

    # -- Category 2: Precondition violations --
    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_fft.fft(None)

    def test_require_non_empty(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_fft.fft([])

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        a = [1, 2, 3, 4]
        res = ag_fft.fft(a, n=8)
        assert res.shape == (8,)

    # -- Category 4: Edge cases --
    def test_1d_complex(self):
        a = np.array([1+1j, 2+2j])
        res = ag_fft.fft(a)
        assert res.dtype == np.complex128

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        a = np.random.rand(8)
        res_atom = ag_fft.fft(a, n=10)
        res_raw = np.fft.fft(a, n=10)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestIfft:
    """Tests for the ifft atom."""

    def test_positive_basic(self):
        a = [10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]
        res = ag_fft.ifft(a)
        expected = np.fft.ifft(a)
        assert np.allclose(res, expected)

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_fft.ifft(None)

    def test_postcondition_shape(self):
        a = [1, 2, 3, 4]
        res = ag_fft.ifft(a)
        assert res.shape == (4,)

    def test_reconstruction(self):
        a = [1, 2, 3, 4]
        f = ag_fft.fft(a)
        inv = ag_fft.ifft(f)
        assert np.allclose(inv, a)

    def test_matches_upstream(self):
        a = np.random.rand(4) + 1j * np.random.rand(4)
        res_atom = ag_fft.ifft(a)
        res_raw = np.fft.ifft(a)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestFftfreq:
    """Tests for the fftfreq atom."""

    def test_positive_basic(self):
        res = ag_fft.fftfreq(4, d=1.0)
        expected = [0. ,  0.25, -0.5 , -0.25]
        assert np.allclose(res, expected)

    def test_require_non_negative(self):
        with pytest.raises(icontract.ViolationError, match="positive"):
            ag_fft.fftfreq(0)
        with pytest.raises(icontract.ViolationError, match="positive"):
            ag_fft.fftfreq(8, d=0)

    def test_postcondition_shape(self):
        res = ag_fft.fftfreq(10)
        assert res.shape == (10,)

    def test_n_one(self):
        res = ag_fft.fftfreq(1)
        assert res.size == 1
        assert res[0] == 0

    def test_matches_upstream(self):
        res_atom = ag_fft.fftfreq(8, d=0.1)
        res_raw = np.fft.fftfreq(8, d=0.1)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestFftshift:
    """Tests for the fftshift atom."""

    def test_positive_basic(self):
        x = [0, 1, 2, 3, 4]
        res = ag_fft.fftshift(x)
        expected = [3, 4, 0, 1, 2]
        assert np.allclose(res, expected)

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_fft.fftshift(None)

    def test_postcondition_shape(self):
        x = np.random.rand(2, 3)
        res = ag_fft.fftshift(x)
        assert res.shape == (2, 3)

    def test_2d_shift(self):
        x = [[1, 2], [3, 4]]
        res = ag_fft.fftshift(x)
        expected = [[4, 3], [2, 1]] # fftshift swaps quadrants
        # Actually np.fft.fftshift([[1, 2], [3, 4]]) -> [[4, 3], [2, 1]]
        assert np.allclose(res, [[4, 3], [2, 1]])

    def test_matches_upstream(self):
        x = np.random.rand(4, 4)
        res_atom = ag_fft.fftshift(x)
        res_raw = np.fft.fftshift(x)
        np.testing.assert_array_equal(res_atom, res_raw)
