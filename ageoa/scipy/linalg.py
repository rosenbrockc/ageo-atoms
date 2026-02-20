import numpy as np
import scipy.linalg
import icontract
from typing import Union, Any, Tuple

# Types
ArrayLike = Union[np.ndarray, list, tuple]

def _is_square_2d(a: ArrayLike) -> bool:
    """Check that a is a 2D square matrix."""
    a_arr = np.asarray(a)
    return a_arr.ndim == 2 and a_arr.shape[0] == a_arr.shape[1]

@icontract.require(lambda a, b: np.asarray(a).ndim == 2, "a must be a 2D matrix")
@icontract.require(lambda a, b: _is_square_2d(a), "a must be square")
@icontract.require(lambda a, b: np.asarray(a).shape[0] == np.asarray(b).shape[0], "Dimensions of a and b must match")
@icontract.ensure(lambda result, a, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def solve(
    a: ArrayLike,
    b: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str = "gen",
) -> np.ndarray:
    """Solve the linear equation a @ x == b for x.

    Args:
        a: Coefficient matrix, shape (n, n).
        b: Ordinate values, shape (n,) or (n, k).
        lower: Use only data contained in the lower triangle of a.
            Default is to use upper triangle.
        overwrite_a: Allow overwriting data in a (may enhance
            performance).
        overwrite_b: Allow overwriting data in b (may enhance
            performance).
        check_finite: Whether to check that the input matrices contain
            only finite numbers.
        assume_a: Type of data in a. Default is 'gen' (general).

    Returns:
        Solution to the system a @ x == b. Shape matches b.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Linear Constraint System Resolver",
        "conceptual_transform": "Finds the unique state configuration that satisfies a set of linear constraints defined by a transformation matrix and a target observation vector. It effectively calculates the pre-image of a vector under a specific linear mapping.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 2D square tensor representing the linear transformation or constraint matrix."
            },
            {
                "name": "b",
                "description": "A 1D or 2D tensor representing the target state or boundary conditions."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor of the same shape as the target, representing the resolved state configuration."
            }
        ],
        "algorithmic_properties": [
            "linear-algebraic",
            "exact-solution",
            "matrix-inversion-equivalent"
        ],
        "cross_disciplinary_applications": [
            "Finding equilibrium currents in a resistor network.",
            "Solving steady-state temperature in heat transfer models.",
            "Determining optimal resource allocation weights under linear equality constraints."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.linalg.solve(
        a,
        b,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        assume_a=assume_a,
    )

@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result.shape == np.asarray(a).shape, "Inverse has same shape as input")
def inv(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """Compute the inverse of a matrix.

    Args:
        a: Square matrix to be inverted.
        overwrite_a: Allow overwriting data in a (may enhance
            performance).
        check_finite: Whether to check that the input matrix contains
            only finite numbers.

    Returns:
        Inverse of the matrix a.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Linear Operator Inverter",
        "conceptual_transform": "Computes the unique linear operator that perfectly reverses the effect of a given full-rank square linear mapping.",
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
            "Reversing 3D spatial transformations in robotics.",
            "Computing precision matrices in multivariate statistics.",
            "Undoing interference in communication channels."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.linalg.inv(a, overwrite_a=overwrite_a, check_finite=check_finite)

@icontract.require(lambda a: np.asarray(a).ndim >= 2, "a must have at least 2 dimensions")
@icontract.require(lambda a: np.asarray(a).shape[-1] == np.asarray(a).shape[-2], "Last two dimensions of a must be square")
@icontract.ensure(lambda result: result is not None, "Determinant must not be None")
def det(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> float:
    """Compute the determinant of a matrix.

    Args:
        a: Square matrix, shape (..., M, M), of which the determinant
            is computed.
        overwrite_a: Allow overwriting data in a (may enhance
            performance).
        check_finite: Whether to check that the input matrix contains
            only finite numbers.

    Returns:
        Determinant of a.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "N-Dimensional Volume Scaling Factor",
        "conceptual_transform": "Calculates the scalar representing the volume expansion or contraction factor of a linear transformation, and whether it preserves or reverses orientation.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A tensor with at least 2 dimensions where the last two are square, representing a linear operator."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A scalar representing the volume scaling factor."
            }
        ],
        "algorithmic_properties": [
            "linear-algebraic",
            "scalar-reduction",
            "volume-measure"
        ],
        "cross_disciplinary_applications": [
            "Checking for singular configurations in robotic arm kinematics.",
            "Normalizing probability density functions.",
            "Determining unique solution existence in control theory."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return float(scipy.linalg.det(a, overwrite_a=overwrite_a, check_finite=check_finite))

@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result[0].shape == np.asarray(a).shape, "LU factor has same shape as input")
@icontract.ensure(lambda result, a: result[1].shape == (np.asarray(a).shape[0],), "Pivot array has length n")
def lu_factor(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pivoted LU decomposition of a square matrix.

    The decomposition satisfies A = P @ L @ U where P is a permutation
    matrix derived from the pivot indices.

    Args:
        a: Square matrix to decompose, shape (n, n).
        overwrite_a: Whether to overwrite data in a (may improve
            performance).
        check_finite: Whether to check that the input contains only
            finite numbers.

    Returns:
        Tuple of (lu, piv) where lu is the LU factor matrix of shape
        (n, n) and piv is the pivot index array of shape (n,).
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Pivoted Linear Factorization Decomposer",
        "conceptual_transform": "Decomposes a square linear operator into a product of lower and upper triangular structures, incorporating a permutation to ensure numerical stability. This provides an intermediate representation for efficient system resolution and determinant calculation.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 2D square tensor representing the linear operator."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple containing the combined triangular factors and the permutation indices."
            }
        ],
        "algorithmic_properties": [
            "matrix-factorization",
            "numerical-stability-optimized",
            "intermediate-representation"
        ],
        "cross_disciplinary_applications": [
            "Preprocessing complex structural matrices for rapid stress analysis.",
            "Efficiently computing determinants of large-scale systems.",
            "Solving multiple systems with the same coefficient structure."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.linalg.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)

@icontract.require(lambda lu_and_piv, b: len(lu_and_piv) == 2, "lu_and_piv must be a tuple of (lu, piv)")
@icontract.require(lambda lu_and_piv, b: lu_and_piv[0].shape[0] == np.asarray(b).shape[0], "Dimensions of LU and b must match")
@icontract.ensure(lambda result, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def lu_solve(
    lu_and_piv: Tuple[np.ndarray, np.ndarray],
    b: ArrayLike,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """Solve an equation system, a @ x = b, given the LU factorization of a.

    Args:
        lu_and_piv: Factorization of the coefficient matrix a, as given
            by lu_factor.
        b: Right-hand side.
        trans: Type of system to solve: 0: a @ x = b, 1: a^T @ x = b,
            2: a^H @ x = b.
        overwrite_b: Whether to overwrite data in b (may improve
            performance).
        check_finite: Whether to check that the input contains only
            finite numbers.

    Returns:
        Solution to the system a @ x = b.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Factorized Linear System Resolver",
        "conceptual_transform": "Efficiently resolves a linear constraint system using a pre-computed triangular factorization. It leverages the simplified structure of the factorized operator to find the solution via forward and backward substitution.",
        "abstract_inputs": [
            {
                "name": "lu_and_piv",
                "description": "A tuple containing the pre-computed triangular factors and permutation indices."
            },
            {
                "name": "b",
                "description": "A 1D or 2D tensor representing the target state."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor representing the resolved state configuration."
            }
        ],
        "algorithmic_properties": [
            "linear-algebraic",
            "substitution-based",
            "efficiency-optimized"
        ],
        "cross_disciplinary_applications": [
            "Rapidly solving for time-varying loads in mechanical systems.",
            "Processing real-time signals through a fixed linear filter.",
            "Updating state estimates in an iterative optimization loop."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.linalg.lu_solve(
        lu_and_piv,
        b,
        trans=trans,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )
