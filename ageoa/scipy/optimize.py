import numpy as np
import scipy.optimize
import icontract
from typing import Union, Any, Callable, Sequence, Dict, Optional, Tuple
from ageoa.ghost.registry import register_atom
from ageoa.scipy.witnesses import (
    witness_scipy_curve_fit,
    witness_scipy_linprog,
    witness_scipy_minimize,
    witness_scipy_root,
)

# Types
ArrayLike = Union[np.ndarray, list, tuple]

@register_atom(witness_scipy_minimize, name="scipy.optimize.minimize")
@icontract.require(lambda x0: np.asarray(x0).ndim >= 1, "Initial guess x0 must be at least 1D")
@icontract.require(lambda fun, x0: fun is not None and x0 is not None, "Objective function and initial guess must not be None")
@icontract.ensure(lambda result: result is not None, "Optimization result must not be None")
def minimize(
    fun: Callable,
    x0: ArrayLike,
    args: tuple = (),
    method: str | None = None,
    jac: Callable | str | bool | None = None,
    hess: Callable | str | None = None,
    hessp: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Dict | Sequence[Dict] = (),
    tol: float | None = None,
    callback: Callable | None = None,
    options: Dict | None = None,
) -> scipy.optimize.OptimizeResult:
    """Minimization of scalar function of one or more variables.

    Args:
        fun: The objective function to be minimized.
        x0: Initial guess.
        args: Extra arguments passed to the objective function.
        method: Type of solver.
        jac: Method for computing the Jacobian (matrix of first partial
            derivatives), used here as the gradient vector.
        hess: Method for computing the Hessian (matrix of second partial
            derivatives) matrix.
        hessp: Hessian of objective function times an arbitrary vector p.
        bounds: Bounds on variables.
        constraints: Constraints definition.
        tol: Tolerance for termination.
        callback: Called after each iteration.
        options: A dictionary of solver options.

    Returns:
        The optimization result represented as a ``OptimizeResult`` object.
    
    """
    return scipy.optimize.minimize(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        callback=callback,
        options=options,
    )

@register_atom(witness_scipy_root, name="scipy.optimize.root")
@icontract.require(lambda fun, x0: fun is not None and x0 is not None, "Function and initial guess must not be None")
@icontract.ensure(lambda result: result is not None, "Root finding result must not be None")
def root(
    fun: Callable,
    x0: ArrayLike,
    args: tuple = (),
    method: str = "hybr",
    jac: Callable | bool | None = None,
    tol: float | None = None,
    callback: Callable | None = None,
    options: Dict | None = None,
) -> scipy.optimize.OptimizeResult:
    """Find a root of a vector function.

    Args:
        fun: Vector function to find a root of.
        x0: Initial guess.
        args: Extra arguments passed to the objective function and its
            Jacobian.
        method: Type of solver.
        jac: Method for computing the Jacobian (matrix of first partial
            derivatives).
        tol: Tolerance for termination.
        callback: Optional callback function.
        options: A dictionary of solver options.

    Returns:
        The solution represented as a ``OptimizeResult`` object.
    
    """
    return scipy.optimize.root(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac,
        tol=tol,
        callback=callback,
        options=options,
    )

@register_atom(witness_scipy_linprog, name="scipy.optimize.linprog")
@icontract.require(lambda c: np.asarray(c).ndim >= 1, "Objective coefficients c must be at least 1D")
@icontract.require(lambda c: c is not None, "Coefficients of the linear objective function must not be None")
@icontract.ensure(lambda result: result is not None, "Linear programming result must not be None")
def linprog(
    c: ArrayLike,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: Sequence | None = None,
    method: str = "highs",
    callback: Callable | None = None,
    options: Dict | None = None,
    x0: ArrayLike | None = None,
) -> scipy.optimize.OptimizeResult:
    """Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints.

    Args:
        c: The coefficients of the linear objective function to be minimized.
        A_ub: The inequality constraint matrix.
        b_ub: The inequality constraint vector.
        A_eq: The equality constraint matrix.
        b_eq: The equality constraint vector.
        bounds: A sequence of ``(min, max)`` pairs for each element in x.
        method: The algorithm used to solve the standard form problem.
        callback: If a callback function is provided, it will be called
            at least once per iteration of the algorithm.
        options: A dictionary of solver options.
        x0: Initial guess of the optimal solution.

    Returns:
        A ``scipy.optimize.OptimizeResult`` consisting of the following
        fields: x, fun, success, slack, con, status, message, nit.
    
    """
    return scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        callback=callback,
        options=options,
        x0=x0,
    )

@register_atom(witness_scipy_curve_fit, name="scipy.optimize.curve_fit")
@icontract.require(lambda f, xdata, ydata: len(xdata) == len(ydata), "xdata and ydata must have the same length")
@icontract.ensure(lambda result: len(result) == 2, "Result must be a tuple of (popt, pcov)")
def curve_fit(
    f: Callable,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    check_finite: bool | None = None,
    bounds: Sequence | None = (-np.inf, np.inf),
    method: str | None = None,
    jac: Callable | str | None = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use non-linear least squares to fit a function, f, to data.

    Args:
        f: The model function, f(x, ...).
        xdata: The independent variable where the data is measured.
        ydata: The dependent data, a length M array - nominally f(xdata, ...).
        p0: Initial guess for the parameters.
        sigma: Determines the uncertainty in ydata.
        absolute_sigma: If True, sigma is used in an absolute sense
            and the estimated parameter covariance pcov reflects these
            absolute values.
        check_finite: If True, check that the input arrays do not contain
            nans of infs.
        bounds: Lower and upper bounds on parameters.
        method: Method to use for optimization.
        jac: Function with signature jac(x, ...) which computes the
            Jacobian (matrix of first partial derivatives) of the model
            function with respect to parameters as a dense array_like
            structure.

    Returns:
        popt: Optimal values for the parameters so that the sum of the
            squared residuals of f(xdata, *popt) - ydata is minimized.
        pcov: The estimated covariance of popt.
    
    """
    return scipy.optimize.curve_fit(
        f,
        xdata,
        ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        bounds=bounds,
        method=method,
        jac=jac,
        **kwargs,
    )
