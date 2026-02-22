"""Tests for ageoa.scipy.stats atoms."""

import numpy as np
import pytest
import icontract
import scipy.stats
from ageoa.scipy import stats as ag_stats

class TestDescribe:
    """Tests for the describe atom."""

    def test_positive_basic(self):
        data = [1, 2, 3, 4, 5]
        res = ag_stats.describe(data)
        assert res.nobs == 5
        assert res.mean == 3.0

    def test_require_non_empty(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_stats.describe([])

    def test_require_non_none(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_stats.describe(None)

    def test_edge_case_single_element(self):
        res = ag_stats.describe([42.0])
        assert res.nobs == 1
        assert res.mean == 42.0

    def test_edge_case_constant_array(self):
        res = ag_stats.describe([7.0, 7.0, 7.0])
        assert res.mean == 7.0
        assert res.variance == 0.0

    def test_upstream_parity(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected = scipy.stats.describe(data)
        actual = ag_stats.describe(data)
        assert actual.nobs == expected.nobs
        assert np.allclose(actual.mean, expected.mean)
        assert np.allclose(actual.variance, expected.variance)

class TestTtestInd:
    """Tests for the ttest_ind atom."""

    def test_positive_basic(self):
        rvs1 = [1, 2, 3, 4, 5]
        rvs2 = [10, 11, 12, 13, 14]
        res = ag_stats.ttest_ind(rvs1, rvs2)
        assert res.pvalue < 0.05

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_stats.ttest_ind(None, [1])

    def test_upstream_parity(self):
        a = [1, 2, 3, 4, 5]
        b = [2, 3, 4, 5, 6]
        expected = scipy.stats.ttest_ind(a, b)
        actual = ag_stats.ttest_ind(a, b)
        assert np.allclose(actual.statistic, expected.statistic)
        assert np.allclose(actual.pvalue, expected.pvalue)

class TestPearsonr:
    """Tests for the pearsonr atom."""

    def test_positive_basic(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        res = ag_stats.pearsonr(x, y)
        assert np.allclose(res.statistic, 1.0)

    def test_require_same_length(self):
        with pytest.raises(icontract.ViolationError, match="same length"):
            ag_stats.pearsonr([1, 2], [1])

    def test_require_min_obs(self):
        with pytest.raises(icontract.ViolationError, match="at least two"):
            ag_stats.pearsonr([1], [1])

    def test_upstream_parity(self):
        x = [1, 3, 5, 7, 9]
        y = [2, 6, 10, 14, 18]
        expected = scipy.stats.pearsonr(x, y)
        actual = ag_stats.pearsonr(x, y)
        assert np.allclose(actual.statistic, expected.statistic)

class TestNorm:
    """Tests for the norm atom."""

    def test_positive_basic(self):
        dist = ag_stats.norm(loc=0, scale=1)
        assert np.allclose(dist.pdf(0), 1/np.sqrt(2*np.pi))

    def test_require_positive_scale(self):
        with pytest.raises(icontract.ViolationError, match="positive"):
            ag_stats.norm(scale=0)

    def test_upstream_parity(self):
        expected_dist = scipy.stats.norm(loc=5, scale=2)
        actual_dist = ag_stats.norm(loc=5, scale=2)
        x_vals = np.linspace(-5, 15, 50)
        assert np.allclose(actual_dist.pdf(x_vals), expected_dist.pdf(x_vals))
