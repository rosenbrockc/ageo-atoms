import numpy as np
import numpy.polynomial.polynomial as poly
import icontract
from typing import Sequence, Union, Any, Tuple
from ageoa.ghost.registry import register_atom
from ageoa.numpy.witnesses import (
    witness_np_polyadd,
    witness_np_polyder,
    witness_np_polyfit,
    witness_np_polyint,
    witness_np_polymul,
    witness_np_polyroots,
    witness_np_polyval,
)

# Types
ArrayLike = Union[np.ndarray, list, tuple]
CoefficientLike = Union[np.ndarray, list, tuple]

@register_atom(witness_np_polyval, name="numpy.polynomial.polyval")
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
    
    """
    return poly.polyval(x, c)

@register_atom(witness_np_polyfit, name="numpy.polynomial.polyfit")
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
    
    """
    return poly.polyfit(x, y, deg)

@register_atom(witness_np_polyder, name="numpy.polynomial.polyder")
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
    
    """
    return poly.polyder(c, m=m)

@register_atom(witness_np_polyint, name="numpy.polynomial.polyint")
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
    
    """
    return poly.polyint(c, m=m, k=k)

@register_atom(witness_np_polyadd, name="numpy.polynomial.polyadd")
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
    
    """
    return poly.polyadd(c1, c2)

@register_atom(witness_np_polymul, name="numpy.polynomial.polymul")
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
    
    """
    return poly.polymul(c1, c2)

@register_atom(witness_np_polyroots, name="numpy.polynomial.polyroots")
@icontract.require(lambda c: len(np.asarray(c)) >= 2, "Polynomial must have at least degree 1 to have roots")
@icontract.ensure(lambda result, c: len(result) == len(c) - 1, "Number of roots must match polynomial degree")
def polyroots(c: CoefficientLike) -> np.ndarray:
    """Compute the roots of a polynomial.

    Args:
        c: Array of coefficients ordered from low to high.

    Returns:
        Roots of the polynomial.
    
    """
    return poly.polyroots(c)
