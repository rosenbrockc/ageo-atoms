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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Linear System Resolver",
        "conceptual_transform": "Finds the exact state configuration that maps to a given observation through a known linear operator, effectively computing the pre-image of a vector under a matrix transformation.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 2D square tensor representing the linear operator or constraint graph."
            },
            {
                "name": "b",
                "description": "A 1D or 2D tensor representing the target observation or boundary conditions."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor of the same shape as the target representing the resolved state configuration."
            }
        ],
        "algorithmic_properties": [
            "linear-algebraic",
            "exact-solution",
            "matrix-inversion-equivalent"
        ],
        "cross_disciplinary_applications": [
            "Finding equilibrium currents in a complex resistor network.",
            "Solving steady-state temperature distributions in heat transfer models.",
            "Determining optimal resource allocation weights under linear equality constraints."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Linear Operator Inverter",
        "conceptual_transform": "Computes the unique linear operator that perfectly reverses the effect of a given full-rank square linear operator, transforming an affine mapping into its inverse mapping.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 2D square tensor representing the forward linear operator."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 2D square tensor representing the inverse linear operator."
            }
        ],
        "algorithmic_properties": [
            "linear-algebraic",
            "matrix-inversion",
            "reversible"
        ],
        "cross_disciplinary_applications": [
            "Reversing 3D spatial transformations in computer graphics and robotics.",
            "Computing the precision matrix from a covariance matrix in multivariate statistics.",
            "Undoing cross-talk interference between parallel communication channels."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "N-Dimensional Volume Scaling Factor",
        "conceptual_transform": "Calculates the scalar value representing how much a given linear operator expands or contracts the volume of an N-dimensional region, and whether it preserves or reverses orientation.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A tensor with at least 2 dimensions where the last two are square, representing a linear operator or a batch of linear operators."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A scalar or an array of scalars representing the volume scaling factor(s)."
            }
        ],
        "algorithmic_properties": [
            "linear-algebraic",
            "scalar-reduction",
            "volume-measure"
        ],
        "cross_disciplinary_applications": [
            "Checking for singular configurations in robotic arm kinematics (Jacobian determinant).",
            "Normalizing probability density functions of multivariate Gaussian distributions.",
            "Determining if a system of equations has a unique solution in control theory."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Multi-Dimensional Magnitude Metric",
        "conceptual_transform": "Reduces a tensor (or specific dimensions of a tensor) into a single non-negative scalar value that quantifies its overall 'size', 'length', or 'magnitude' according to a specified distance metric.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "An N-dimensional tensor whose magnitude is to be measured."
            },
            {
                "name": "ord",
                "description": "A parameter defining the specific distance metric (e.g., Euclidean, Manhattan, maximum absolute value)."
            },
            {
                "name": "axis",
                "description": "An optional integer or tuple specifying which dimensions to reduce."
            },
            {
                "name": "keepdims",
                "description": "A boolean indicating whether to retain the reduced dimensions as size 1."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A non-negative scalar or tensor of magnitudes."
            }
        ],
        "algorithmic_properties": [
            "reduction",
            "distance-metric",
            "non-negative"
        ],
        "cross_disciplinary_applications": [
            "Measuring the error between predicted and actual values in machine learning (Loss functions).",
            "Normalizing term frequency vectors in natural language processing (TF-IDF).",
            "Quantifying the total energy of a wave function in quantum mechanics."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
