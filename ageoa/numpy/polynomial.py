import numpy as np
import numpy.polynomial.polynomial as poly
import icontract
from typing import Sequence, Union, Any, Tuple

# Types
ArrayLike = Union[np.ndarray, list, tuple]
CoefficientLike = Union[np.ndarray, list, tuple]

@icontract.require(lambda c, x: c is not None and x is not None, "Coefficients and x must not be None")
@icontract.require(lambda c: len(np.asarray(c)) > 0, "Coefficients must not be empty")
@icontract.ensure(lambda result, x: np.asarray(result).shape == np.asarray(x).shape, "Result shape must match x shape")
def polyval(x: Any, c: CoefficientLike) -> Any:
    """Evaluate a polynomial at points x.

    If c is of length n + 1, this function returns the value
    p(x) = c[0] + c[1]*x + ... + c[n]*x^n

    Args:
        x: Points at which to evaluate the polynomial.
        c: Array of coefficients ordered from low to high.

    Returns:
        The values of the polynomial at points x.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Rational Function Basis Evaluator",
        "conceptual_transform": "Computes the output of a model defined as a linear combination of power-basis functions (1, x, x^2, ...) at specified evaluation points. It maps a parameter vector (coefficients) to a set of outputs based on input coordinates.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A tensor of coordinates at which to evaluate the model."
            },
            {
                "name": "c",
                "description": "A 1D tensor of weights (coefficients) for each power-basis function."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor of the same shape as the coordinates containing the model outputs."
            }
        ],
        "algorithmic_properties": [
            "linear-combination",
            "power-basis",
            "deterministic",
            "model-evaluation"
        ],
        "cross_disciplinary_applications": [
            "Predicting future values from a fitted growth model in biology.",
            "Evaluating a calibrated calibration curve in analytical chemistry.",
            "Calculating the trajectory of a projectile from its motion coefficients."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return poly.polyval(x, c)

@icontract.require(lambda x, y, deg: len(np.asarray(x)) == len(np.asarray(y)), "x and y must have the same length")
@icontract.require(lambda deg: deg >= 0, "Degree must be non-negative")
@icontract.ensure(lambda result, deg: len(result) == deg + 1, "Result must have deg + 1 coefficients")
def polyfit(x: ArrayLike, y: ArrayLike, deg: int) -> np.ndarray:
    """Least-squares fit of a polynomial to data.

    Returns the coefficients of a polynomial of degree deg that is the
    least squares fit to the data values y given at points x.

    Args:
        x: x-coordinates of the M sample points.
        y: y-coordinates of the M sample points.
        deg: Degree of the fitting polynomial.

    Returns:
        Polynomial coefficients ordered from low to high.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Power-Basis Least-Squares Optimizer",
        "conceptual_transform": "Finds the optimal set of weights for a power-basis expansion that minimizes the sum of squared residuals relative to a provided set of observation pairs. It performs a linear regression in a higher-dimensional feature space.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A 1D tensor of independent variable observations."
            },
            {
                "name": "y",
                "description": "A 1D tensor of dependent variable observations."
            },
            {
                "name": "deg",
                "description": "An integer specifying the maximum power to include in the basis (model complexity)."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of optimized weights ordered from lowest to highest power."
            }
        ],
        "algorithmic_properties": [
            "least-squares",
            "linear-regression",
            "parameter-optimization",
            "model-fitting"
        ],
        "cross_disciplinary_applications": [
            "Fitting a smooth trend curve to noisy sequential measurements from a physical sensor.",
            "Calibrating a sensor by finding the best polynomial mapping from raw voltage to physical units.",
            "Characterizing the non-linear response of a material under increasing stress."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return poly.polyfit(x, y, deg)

@icontract.require(lambda c: len(np.asarray(c)) > 0, "Coefficients must not be empty")
@icontract.ensure(lambda result, c, m: len(result) == max(1, len(c) - m), "Result length must be correct after differentiation")
def polyder(c: CoefficientLike, m: int = 1) -> np.ndarray:
    """Differentiate a polynomial.

    Returns the coefficients of the derivative of the polynomial c.

    Args:
        c: Array of coefficients ordered from low to high.
        m: Number of derivatives to take, must be non-negative.

    Returns:
        Coefficients of the derivative.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Polynomial Operator Differentiator",
        "conceptual_transform": "Transforms the coefficients of a power-basis expansion to represent its instantaneous rate of change (derivative) with respect to the independent variable. It performs a linear shift and scaling in the coefficient space.",
        "abstract_inputs": [
            {
                "name": "c",
                "description": "A 1D tensor of weights for the original model."
            },
            {
                "name": "m",
                "description": "An integer specifying the number of successive differentiations to perform."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of weights for the derivative model."
            }
        ],
        "algorithmic_properties": [
            "linear-operator",
            "calculus-derivative",
            "structural-transform"
        ],
        "cross_disciplinary_applications": [
            "Calculating velocity and acceleration from a position-time polynomial model.",
            "Computing the instantaneous rate of change from a polynomial model of cumulative throughput.",
            "Analyzing the slope of a fitted calibration curve for sensitivity analysis."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return poly.polyder(c, m=m)

@icontract.require(lambda c: len(np.asarray(c)) > 0, "Coefficients must not be empty")
@icontract.require(lambda m, k: (len(k) if np.iterable(k) else 1) <= m, "Too many integration constants")
@icontract.ensure(lambda result, c, m: len(result) == len(c) + m, "Result length must be correct after integration")
def polyint(c: CoefficientLike, m: int = 1, k: ArrayLike | float = 0) -> np.ndarray:
    """Integrate a polynomial.

    Returns the coefficients of the integral of the polynomial c.

    Args:
        c: Array of coefficients ordered from low to high.
        m: Order of integration, must be positive.
        k: Integration constant(s).

    Returns:
        Coefficients of the integral.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Polynomial Operator Integrator",
        "conceptual_transform": "Transforms the coefficients of a power-basis expansion to represent its accumulation (integral) with respect to the independent variable. It introduces a constant of integration and performs a linear shift/scaling in the coefficient space.",
        "abstract_inputs": [
            {
                "name": "c",
                "description": "A 1D tensor of weights for the original model."
            },
            {
                "name": "m",
                "description": "An integer specifying the number of successive integrations to perform."
            },
            {
                "name": "k",
                "description": "A scalar or array defining the integration constants (boundary conditions)."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of weights for the integrated model."
            }
        ],
        "algorithmic_properties": [
            "linear-operator",
            "calculus-integral",
            "structural-transform",
            "boundary-condition-dependent"
        ],
        "cross_disciplinary_applications": [
            "Calculating total distance traveled from an acceleration-time polynomial model.",
            "Estimating total resource accumulation from a rate-of-consumption model.",
            "Finding the potential energy field from a polynomial force-field representation."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return poly.polyint(c, m=m, k=k)

@icontract.require(lambda c1, c2: True, "Placeholder for polyadd")
@icontract.ensure(lambda result, c1, c2: len(result) == max(len(c1), len(c2)), "Result length must match maximum of input lengths")
def polyadd(c1: CoefficientLike, c2: CoefficientLike) -> np.ndarray:
    """Add one polynomial to another.

    Returns the sum of two polynomials c1 + c2.

    Args:
        c1: Array of coefficients of the first polynomial.
        c2: Array of coefficients of the second polynomial.

    Returns:
        Coefficients of the sum.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Power-Basis Model Adder",
        "conceptual_transform": "Computes the coefficients of a new model that is the linear sum of two existing power-basis models. It performs an element-wise addition of weights corresponding to the same basis functions.",
        "abstract_inputs": [
            {
                "name": "c1",
                "description": "Weights for the first model."
            },
            {
                "name": "c2",
                "description": "Weights for the second model."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "Weights for the summed model."
            }
        ],
        "algorithmic_properties": [
            "linear-operation",
            "additive-synthesis"
        ],
        "cross_disciplinary_applications": [
            "Combining multiple independent physical influences into a single superposed model.",
            "Aggregating sub-models in a multi-component statistical analysis.",
            "Superimposing different noise and trend components in a time-series model."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return poly.polyadd(c1, c2)

@icontract.require(lambda c1, c2: True, "Placeholder for polymul")
@icontract.ensure(lambda result, c1, c2: len(result) == len(c1) + len(c2) - 1 if len(c1) > 0 and len(c2) > 0 else 0, "Result length must match product of input lengths")
def polymul(c1: CoefficientLike, c2: CoefficientLike) -> np.ndarray:
    """Multiply one polynomial by another.

    Returns the product of two polynomials c1 * c2.

    Args:
        c1: Array of coefficients of the first polynomial.
        c2: Array of coefficients of the second polynomial.

    Returns:
        Coefficients of the product.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Power-Basis Model Convolver",
        "conceptual_transform": "Computes the coefficients of a new model that is the product of two existing power-basis models. It performs a discrete convolution of the weight vectors, representing the interaction of all basis terms.",
        "abstract_inputs": [
            {
                "name": "c1",
                "description": "Weights for the first model."
            },
            {
                "name": "c2",
                "description": "Weights for the second model."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "Weights for the product model."
            }
        ],
        "algorithmic_properties": [
            "convolutional",
            "multiplicative-interaction",
            "structural-transform"
        ],
        "cross_disciplinary_applications": [
            "Modeling the interaction of multiple independent factors in a physical system.",
            "Constructing high-order filters from simpler second-order stages in the coefficient domain.",
            "Calculating the joint distribution of independent random variables in probability theory."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return poly.polymul(c1, c2)

@icontract.require(lambda c: len(np.asarray(c)) >= 2, "Polynomial must have at least degree 1 to have roots")
@icontract.ensure(lambda result, c: len(result) == len(c) - 1, "Number of roots must match polynomial degree")
def polyroots(c: CoefficientLike) -> np.ndarray:
    """Compute the roots of a polynomial.

    Args:
        c: Array of coefficients ordered from low to high.

    Returns:
        Roots of the polynomial.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Model Zero-Crossing Finder",
        "conceptual_transform": "Identifies the input coordinates where the output of a power-basis model is exactly zero. It solves for the fundamental equilibrium points of the model representation.",
        "abstract_inputs": [
            {
                "name": "c",
                "description": "A 1D tensor of weights defining the model."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of coordinates (potentially complex) where the model evaluates to zero."
            }
        ],
        "algorithmic_properties": [
            "root-finding",
            "eigenvalue-equivalent",
            "non-linear-solver"
        ],
        "cross_disciplinary_applications": [
            "Finding the equilibrium points in a dynamic physical system.",
            "Finding the zero-crossing points of a polynomial model describing net energy balance.",
            "Solving for the resonance frequencies in a structural vibration model."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return poly.polyroots(c)
