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

class TestNorm:
    """Tests for the norm atom."""

    def test_positive_basic(self):
        dist = ag_stats.norm(loc=0, scale=1)
        assert np.allclose(dist.pdf(0), 1/np.sqrt(2*np.pi))

    def test_require_positive_scale(self):
        with pytest.raises(icontract.ViolationError, match="positive"):
            ag_stats.norm(scale=0)
