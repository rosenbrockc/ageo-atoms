import numpy as np
import scipy.stats
import icontract
from typing import Union, Any, Tuple, Optional, Sequence

# Types
ArrayLike = Union[np.ndarray, list, tuple]

@icontract.require(lambda a: a is not None, "Input data must not be None")
@icontract.require(lambda a: len(np.asarray(a)) > 0, "Input data must not be empty")
@icontract.ensure(lambda result: result is not None, "Description result must not be None")
def describe(
    a: ArrayLike,
    axis: int | None = 0,
    ddof: int = 1,
    bias: bool = True,
    nan_policy: str = "propagate",
) -> Any:
    """Compute several descriptive statistics of the passed array.

    Args:
        a: Input data.
        axis: Axis along which statistics are calculated. Default is 0.
        ddof: Degrees of freedom adjustment for variance.
        bias: If False, skewness and kurtosis are corrected for
            statistical bias.
        nan_policy: Defines how to handle input NaNs.

    Returns:
        DescribeResult object containing nobs, minmax, mean, variance,
        skewness, and kurtosis.
    """
    return scipy.stats.describe(
        a,
        axis=axis,
        ddof=ddof,
        bias=bias,
        nan_policy=nan_policy,
    )

@icontract.require(lambda a, b: a is not None and b is not None, "Input samples must not be None")
@icontract.ensure(lambda result: len(result) >= 2, "Result must be a tuple of (statistic, pvalue)")
def ttest_ind(
    a: ArrayLike,
    b: ArrayLike,
    axis: int = 0,
    equal_var: bool = True,
    nan_policy: str = "propagate",
    permutations: float | None = None,
    random_state: int | np.random.Generator | np.random.RandomState | None = None,
    alternative: str = "two-sided",
    trim: float = 0,
) -> Any:
    """Calculate the T-test for the means of two independent samples of
    scores.

    Args:
        a, b: The arrays must have the same shape, except in the
            dimension corresponding to axis.
        axis: Axis along which to compute test.
        equal_var: If True, perform a standard independent 2 sample
            test that assumes equal population variances.
        nan_policy: Defines how to handle input NaNs.
        permutations: The number of permutations of the data used to
            estimate p-values.
        random_state: Generator or seed used for permutations.
        alternative: Defines the alternative hypothesis.
        trim: If non-zero, performs a trimmed (Yuen's) t-test.

    Returns:
        Ttest_indResult object with statistic and pvalue.
    """
    return scipy.stats.ttest_ind(
        a,
        b,
        axis=axis,
        equal_var=equal_var,
        nan_policy=nan_policy,
        permutations=permutations,
        random_state=random_state,
        alternative=alternative,
        trim=trim,
    )

@icontract.require(lambda x, y: len(np.asarray(x)) == len(np.asarray(y)), "x and y must have the same length")
@icontract.require(lambda x: len(np.asarray(x)) >= 2, "Need at least two observations")
@icontract.ensure(lambda result: -1 <= result[0] <= 1, "Correlation coefficient must be between -1 and 1")
def pearsonr(x: ArrayLike, y: ArrayLike) -> Any:
    """Pearson correlation coefficient and p-value for testing
    non-correlation.

    Args:
        x: Input array.
        y: Input array.

    Returns:
        PearsonRResult object with statistic and pvalue.
    """
    return scipy.stats.pearsonr(x, y)

@icontract.require(lambda x, y: len(np.asarray(x)) == len(np.asarray(y)), "x and y must have the same length")
@icontract.ensure(lambda result: -1 <= result[0] <= 1, "Correlation coefficient must be between -1 and 1")
def spearmanr(
    a: ArrayLike,
    b: ArrayLike | None = None,
    axis: int | None = 0,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
) -> Any:
    """Calculate a Spearman correlation coefficient with associated
    p-value.

    Args:
        a, b: Two 1-D or 2-D arrays containing samples.
        axis: If axis=0 (default), then each column represents a
            variable.
        nan_policy: Defines how to handle input NaNs.
        alternative: Defines the alternative hypothesis.

    Returns:
        SignificanceResult object with statistic and pvalue.
    """
    return scipy.stats.spearmanr(
        a,
        b=b,
        axis=axis,
        nan_policy=nan_policy,
        alternative=alternative,
    )

@icontract.require(lambda loc, scale: scale > 0, "Scale must be positive")
@icontract.ensure(lambda result: result is not None, "Normal distribution object must not be None")
def norm(loc: float = 0, scale: float = 1) -> Any:
    """A normal continuous random variable.

    Args:
        loc: Mean ("centre") of the distribution.
        scale: Standard deviation of the distribution.

    Returns:
        A frozen normal distribution object.
    """
    return scipy.stats.norm(loc=loc, scale=scale)
