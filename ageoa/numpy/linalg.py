import numpy as np
import icontract
from typing import Any

@icontract.require(lambda a, b: a.ndim == 2, "a must be a 2D matrix")
@icontract.require(lambda a, b: a.shape[0] == a.shape[1], "a must be square")
@icontract.require(lambda a, b: a.shape[0] == b.shape[0], "Dimensions of a and b must match")
def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, x, of the well-determined, i.e., full rank,
    linear matrix equation ax = b.

    Args:
        a: Coefficient matrix.
        b: Ordinate or "dependent variable" values.

    Returns:
        Solution to the system a x = b.
    """
    return np.linalg.solve(a, b)

@icontract.require(lambda a: a.ndim == 2, "a must be a 2D matrix")
@icontract.require(lambda a: a.shape[0] == a.shape[1], "a must be square")
def inv(a: np.ndarray) -> np.ndarray:
    """
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix a, return the matrix a_inv such that dot(a, a_inv) = dot(a_inv, a) = eye(a.shape[0]).

    Args:
        a: Matrix to be inverted.

    Returns:
        Inverse of the matrix a.
    """
    return np.linalg.inv(a)

@icontract.require(lambda a: a.ndim >= 2, "a must have at least 2 dimensions")
@icontract.require(lambda a: a.shape[-1] == a.shape[-2], "The last two dimensions of a must be square")
def det(a: np.ndarray) -> Any:
    """
    Compute the determinant of an array.

    Args:
        a: Input array to compute determinants for.

    Returns:
        Determinant of a.
    """
    return np.linalg.det(a)
