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
        The bases x raised to the probabilities p.
    """
    return np.emath.power(x, p)
