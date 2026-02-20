import numpy as np
import icontract
from typing import Union, Any

# Types
Numeric = Union[float, int, complex, np.number]
ArrayLike = Union[np.ndarray, list, tuple, Numeric]

@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(lambda result, x: np.allclose(np.square(result), x), "Result squared must be approximately x")
def sqrt(x: ArrayLike) -> Any:
    """Compute the square root of x.

    For negative input, the result will be complex.

    Args:
        x: The value(s) whose square root is required.

    Returns:
        The square root of x. If x was a scalar, so is the result,
        otherwise an array is returned.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Geometric Scale Bisector",
        "conceptual_transform": "Computes the value that, when self-interacted via multiplication, yields the original input. It automatically expands the numerical range into the complex domain to handle negative inputs while maintaining functional consistency.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A scalar or tensor of values whose geometric bisector is required."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The computed bisector, potentially complex-valued."
            }
        ],
        "algorithmic_properties": [
            "nonlinear-operator",
            "domain-expanding",
            "invertible-composition"
        ],
        "cross_disciplinary_applications": [
            "Calculating the standard deviation from a variance measurement.",
            "Determining the characteristic length scale of a 2D area.",
            "Computing the magnitude of a 2D vector in Euclidean space."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.emath.sqrt(x)

@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.require(lambda x: np.all(np.asarray(x) != 0), "Logarithm of zero is undefined")
@icontract.ensure(lambda result, x: np.allclose(np.exp(result), x), "Exp of result must be approximately x")
def log(x: ArrayLike) -> Any:
    """Compute the natural logarithm of x.

    Return the "principal value" (for a description of this, see
    `numpy.log`) of $log_e(x)$. For real x < 0, result is complex.

    Args:
        x: The value(s) whose log is required.

    Returns:
        The log of x. If x was a scalar, so is the result,
        otherwise an array is returned.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Natural Exponential Magnitude Resolver",
        "conceptual_transform": "Determines the exponent required to reach a target value from an irrational constant base (e). It maps exponential growth scales to a linear magnitude representation, handling negative target values via complex projection.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A scalar or tensor representing target magnitudes."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The resolved exponential magnitudes (natural logarithms)."
            }
        ],
        "algorithmic_properties": [
            "nonlinear-reduction",
            "domain-expanding",
            "scale-linearization"
        ],
        "cross_disciplinary_applications": [
            "Calculating the half-life of a radioactive decay process.",
            "Measuring the information entropy of a probability distribution.",
            "Analyzing the growth rate of a population under ideal conditions."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.emath.log(x)

@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.require(lambda x: np.all(np.asarray(x) != 0), "Logarithm of zero is undefined")
@icontract.ensure(lambda result, x: np.allclose(np.power(10, result), x), "10 to the power of result must be approximately x")
def log10(x: ArrayLike) -> Any:
    """Compute the logarithm base 10 of x.

    Return the "principal value" of $log_{10}(x)$. For real x < 0,
    result is complex.

    Args:
        x: The value(s) whose log base 10 is required.

    Returns:
        The log base 10 of x. If x was a scalar, so is the result,
        otherwise an array is returned.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Decadal Magnitude Resolver",
        "conceptual_transform": "Maps target values to a power-of-ten scale, providing a linear representation of order-of-magnitude variations. It facilitates the analysis of phenomena spanning multiple scales of intensity.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A scalar or tensor of intensity measurements."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The resolved decadal magnitudes."
            }
        ],
        "algorithmic_properties": [
            "nonlinear-reduction",
            "scale-linearization",
            "order-of-magnitude-mapping"
        ],
        "cross_disciplinary_applications": [
            "Measuring acoustic intensity on a decibel scale.",
            "Analyzing the pH levels of a chemical solution.",
            "Representing the magnitude of seismic events on a Richter-equivalent scale."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.emath.log10(x)

@icontract.require(lambda n, x: n is not None and x is not None, "Base n and value x must not be None")
@icontract.require(lambda n: np.all(np.asarray(n) > 0) and np.all(np.asarray(n) != 1), "Base n must be positive and not equal to 1")
@icontract.require(lambda x: np.all(np.asarray(x) != 0), "Logarithm of zero is undefined")
def logn(n: Numeric, x: ArrayLike) -> Any:
    """Compute the logarithm base n of x.

    Return the "principal value" of $log_n(x)$. For real x < 0,
    result is complex.

    Args:
        n: The base in which the logarithm is taken.
        x: The value(s) whose log base n is required.

    Returns:
        The log base n of x. If x was a scalar, so is the result,
        otherwise an array is returned.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Arbitrary-Base Magnitude Resolver",
        "conceptual_transform": "Determines the linear magnitude of a target value relative to a specified reference growth base. It provides a generalized coordinate system for exponential scaling.",
        "abstract_inputs": [
            {
                "name": "n",
                "description": "A scalar representing the reference growth base."
            },
            {
                "name": "x",
                "description": "A scalar or tensor of target values."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The resolved magnitudes in the specified base."
            }
        ],
        "algorithmic_properties": [
            "nonlinear-reduction",
            "generalized-scaling",
            "coordinate-transform"
        ],
        "cross_disciplinary_applications": [
            "Calculating the depth of a balanced search tree in computer science (Base 2).",
            "Analyzing the number of generations in a binary fission process.",
            "Determining the required steps in a hierarchical decision process."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.emath.logn(n, x)

@icontract.require(lambda x, p: x is not None and p is not None, "Input x and power p must not be None")
def power(x: ArrayLike, p: Any) -> Any:
    """Return x to the power p (x**p).

    If x contains negative values, the output is converted to the
    complex domain.

    Args:
        x: The base.
        p: The exponent(s).

    Returns:
        The bases x raised to the exponents p.

    <!-- conceptual_profile -->
    {
        "abstract_name": "Exponential Growth Operator",
        "conceptual_transform": "Applies repeated self-interaction to a base value according to an exponent, representing non-linear scaling or growth. It automatically handles transitions into the complex domain for negative bases with fractional exponents.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A scalar or tensor representing the base values."
            },
            {
                "name": "p",
                "description": "A scalar or tensor representing the exponential growth factors."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The resulting growth magnitudes."
            }
        ],
        "algorithmic_properties": [
            "nonlinear-transformation",
            "domain-expanding",
            "growth-modeling"
        ],
        "cross_disciplinary_applications": [
            "Modeling exponential amplification or decay in cascaded gain stages.",
            "Modeling the intensity of light attenuation through a medium (Beer-Lambert law).",
            "Calculating the gravitational force between two masses (Inverse-square law)."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.emath.power(x, p)
