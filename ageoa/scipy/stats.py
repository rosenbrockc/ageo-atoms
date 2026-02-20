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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Stochastic Sequence Distribution Summarizer",
        "conceptual_transform": "Computes a set of statistical descriptors that summarize the central tendency, dispersion, and shape of a sample distribution. it maps a raw sequence of observations to a compact representation of its stochastic properties.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A tensor of observations from a stochastic process."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A structured object containing the number of observations, min/max, mean, variance, skewness, and kurtosis."
            }
        ],
        "algorithmic_properties": [
            "statistical-reduction",
            "moment-calculation",
            "summary-statistics"
        ],
        "cross_disciplinary_applications": [
            "Summarizing the performance variability of a manufacturing process from quality control samples.",
            "Characterizing the noise profile of a sensor by analyzing its output distribution over time.",
            "Providing a baseline statistical overview of demographic data in a social science study."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Independent Distribution Parity Tester",
        "conceptual_transform": "Calculates a test statistic and associated probability (p-value) to determine if two independent sample sets are likely to have been drawn from distributions with the same mean. It quantifies the likelihood that the observed difference is due to random chance.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A tensor of observations from the first group."
            },
            {
                "name": "b",
                "description": "A tensor of observations from the second group."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A structured object containing the test statistic and the p-value."
            }
        ],
        "algorithmic_properties": [
            "hypothesis-testing",
            "parametric-test",
            "stochastic-comparison"
        ],
        "cross_disciplinary_applications": [
            "Comparing the central tendency of two independent measurement groups from different experimental conditions.",
            "Evaluating whether a system configuration change produces a statistically significant shift in a scalar performance metric.",
            "Evaluating if two different manufacturing batches have identical physical properties."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Linear Relational Dependency Metric",
        "conceptual_transform": "Quantifies the strength and direction of a linear relationship between two paired sequences of observations. It maps the co-variation of the sequences to a normalized coefficient between -1 and 1.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A 1D tensor of observations for the first variable."
            },
            {
                "name": "y",
                "description": "A 1D tensor of paired observations for the second variable."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An object containing the correlation coefficient and p-value."
            }
        ],
        "algorithmic_properties": [
            "linear-correlation",
            "bivariate-analysis",
            "normalized-metric"
        ],
        "cross_disciplinary_applications": [
            "Quantifying the strength of linear co-variation between two continuously measured physical properties across a sample.",
            "Assessing the degree of linear coupling between an input control variable and an observed response in an experimental system.",
            "Evaluating the linear dependency between two different sensor readings in a monitoring system."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Monotonic Rank Dependency Metric",
        "conceptual_transform": "Quantifies the strength and direction of a monotonic relationship between two paired sequences by analyzing the correlation of their ranks rather than their absolute values. It is robust to non-linearities and outliers.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D tensor of observations for the first variable."
            },
            {
                "name": "b",
                "description": "A 1D tensor of paired observations for the second variable (optional if a is 2D)."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An object containing the rank correlation coefficient and p-value."
            }
        ],
        "algorithmic_properties": [
            "non-parametric-correlation",
            "rank-based",
            "bivariate-analysis",
            "monotonic-dependency"
        ],
        "cross_disciplinary_applications": [
            "Assessing the monotonic association between two ordinal rankings of entities in a survey dataset.",
            "Analyzing the monotonic agreement between two different quality ranking systems.",
            "Detecting monotonic trends between sequential experimental measurements and an ordinal response variable."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Gaussian Probability Density Generator",
        "conceptual_transform": "Returns a representation of a normal (Gaussian) continuous probability distribution with a specified location (mean) and scale (standard deviation). It provides a generative model for stochastic processes governed by the central limit theorem.",
        "abstract_inputs": [
            {
                "name": "loc",
                "description": "A scalar representing the central location (mean) of the distribution."
            },
            {
                "name": "scale",
                "description": "A scalar representing the width (standard deviation) of the distribution."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A frozen distribution object capable of generating samples, computing densities (PDF/CDF), and moments."
            }
        ],
        "algorithmic_properties": [
            "probabilistic-generative",
            "gaussian-model",
            "continuous-distribution"
        ],
        "cross_disciplinary_applications": [
            "Modeling the distribution of measurement errors in a physical experiment.",
            "Generating synthetic samples for Monte Carlo estimation of intractable integrals.",
            "Defining the prior distribution for a Bayesian inference model."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.stats.norm(loc=loc, scale=scale)
