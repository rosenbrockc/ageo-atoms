import numpy as np
import icontract
from typing import Any, Union

# Types
ArrayLike = Union[np.ndarray, list, tuple]

def _is_square_2d(a: ArrayLike) -> bool:
    """Check that a is a 2D square matrix."""
    a_arr = np.asarray(a)
    return a_arr.ndim == 2 and a_arr.shape[0] == a_arr.shape[1]

def _is_square_at_least_2d(a: ArrayLike) -> bool:
    """Check that a has at least 2 dimensions and the last two are square."""
    a_arr = np.asarray(a)
    return a_arr.ndim >= 2 and a_arr.shape[-1] == a_arr.shape[-2]

@icontract.require(lambda a, b: np.asarray(a).ndim == 2, "a must be a 2D matrix")
@icontract.require(lambda a, b: _is_square_2d(a), "a must be square")
@icontract.require(lambda a, b: np.asarray(a).shape[0] == np.asarray(b).shape[0], "Dimensions of a and b must match")
@icontract.ensure(lambda result, a, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, x, of the well-determined, i.e.,
    full rank, linear matrix equation ax = b.

    Args:
        a: Coefficient matrix, shape (n, n). Must be square and
            non-singular.
        b: Ordinate or "dependent variable" values, shape (n,) or (n, k).

    Returns:
        Solution to the system a x = b. Shape matches b.
    
    """
    return np.linalg.solve(a, b)

@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result.shape == np.asarray(a).shape, "Inverse has same shape as input")
def inv(a: np.ndarray) -> np.ndarray:
    """Compute the (multiplicative) inverse of a matrix.

    Given a square matrix a, return the matrix a_inv such that
    dot(a, a_inv) = dot(a_inv, a) = eye(a.shape[0]).

    Args:
        a: Matrix to be inverted, shape (n, n).

    Returns:
        Inverse of the matrix a. Shape matches a.
    
    """
    return np.linalg.inv(a)

@icontract.require(lambda a: np.asarray(a).ndim >= 2, "a must have at least 2 dimensions")
@icontract.require(lambda a: _is_square_at_least_2d(a), "The last two dimensions of a must be square")
@icontract.ensure(lambda result: result is not None, "Determinant must not be None")
def det(a: np.ndarray) -> Any:
    """Compute the determinant of an array.

    Args:
        a: Input array to compute determinants for, shape (..., M, M).

    Returns:
        Determinant of a.
    
    """
    return np.linalg.det(a)

@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(lambda result: np.all(np.asarray(result) >= 0), "Norm must be non-negative")
def norm(
    x: ArrayLike,
    ord: int | float | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Union[float, np.floating, np.ndarray]:
    """Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below),
    depending on the value of the ord parameter.

    Args:
        x: Input array. If axis is None, x must be 1-D or 2-D, unless
            ord is None. If both axis and ord are None, the 2-norm of
            x.ravel will be returned.
        ord: Order of the norm.
        axis: If axis is an integer, it specifies the axis of x along
            which to compute the vector norms. If axis is a 2-tuple,
            it specifies the axes that hold 2-D matrices, and the
            matrix norms of these matrices are computed. If axis is
            None then either a vector norm (when x is 1-D) or a matrix
            norm (when x is 2-D) is returned.
        keepdims: If this is set to True, the axes which are normed
            over are left in the result as dimensions with size one.

    Returns:
        Norm of the matrix or vector.
    
    """
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
