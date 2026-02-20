import numpy as np
import scipy.optimize
import icontract
from typing import Union, Any, Callable, Sequence, Dict, Optional, Tuple

# Types
ArrayLike = Union[np.ndarray, list, tuple]

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
        jac: Method for computing the gradient vector.
        hess: Method for computing the Hessian matrix.
        hessp: Hessian of objective function times an arbitrary vector p.
        bounds: Bounds on variables.
        constraints: Constraints definition.
        tol: Tolerance for termination.
        callback: Called after each iteration.
        options: A dictionary of solver options.

    Returns:
        The optimization result represented as a ``OptimizeResult`` object.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Multi-Dimensional Scalar Function Optimizer",
        "conceptual_transform": "Locates the set of input parameters that minimizes a scalar-valued cost function, potentially subject to boundary constraints and equality/inequality relationships. It performs a guided search in a multi-dimensional parameter space to find an optimal state.",
        "abstract_inputs": [
            {
                "name": "fun",
                "description": "A functional mapping from a parameter vector to a scalar cost."
            },
            {
                "name": "x0",
                "description": "A 1D tensor representing the initial guess for the optimal state."
            },
            {
                "name": "args",
                "description": "Extra static parameters for the cost function."
            },
            {
                "name": "method",
                "description": "A string identifier for the specific numerical optimization algorithm."
            },
            {
                "name": "bounds",
                "description": "Constraints defining the allowable range for each parameter."
            },
            {
                "name": "constraints",
                "description": "Functional or algebraic relationships that the optimal state must satisfy."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A structured object containing the optimized parameter vector and metadata about the optimization process (convergence, final cost, etc.)."
            }
        ],
        "algorithmic_properties": [
            "non-linear-optimization",
            "iterative-search",
            "constraint-aware",
            "gradient-based-capable"
        ],
        "cross_disciplinary_applications": [
            "Calibrating a complex climate model by minimizing the error against historical temperature data.",
            "Optimizing the shape of an airfoil to minimize drag under specified lift constraints.",
            "Finding the lowest-energy configuration of a protein molecule in a force field."
        ]
    }
    <!-- /conceptual_profile -->
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
        jac: Method for computing the Jacobian.
        tol: Tolerance for termination.
        callback: Optional callback function.
        options: A dictionary of solver options.

    Returns:
        The solution represented as a ``OptimizeResult`` object.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Non-Linear Vector Function Equilibrium Finder",
        "conceptual_transform": "Finds the input state where a multi-dimensional vector function evaluates to zero (an equilibrium or null point). It iteratively resolves a system of non-linear equations to find a consistent state.",
        "abstract_inputs": [
            {
                "name": "fun",
                "description": "A functional mapping from a parameter vector to a residual vector of the same dimension."
            },
            {
                "name": "x0",
                "description": "A 1D tensor representing the initial guess for the equilibrium state."
            },
            {
                "name": "method",
                "description": "A string identifier for the specific root-finding algorithm."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A structured object containing the equilibrium state vector."
            }
        ],
        "algorithmic_properties": [
            "non-linear-root-finding",
            "equilibrium-solver",
            "iterative-refinement"
        ],
        "cross_disciplinary_applications": [
            "Solving for steady-state concentrations in a complex chemical reaction network.",
            "Finding equilibrium states in a coupled multi-agent resource exchange model.",
            "Resolving the static balance of forces in a structural engineering truss."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Constrained Linear Objective Resolver",
        "conceptual_transform": "Minimizes a linear objective function subject to a set of linear equality and inequality constraints. It identifies the optimal vertex in a convex polyhedral feasible region defined by linear relationships.",
        "abstract_inputs": [
            {
                "name": "c",
                "description": "A 1D tensor of weights for the linear objective function."
            },
            {
                "name": "A_ub",
                "description": "A 2D tensor representing the coefficients of linear inequality constraints."
            },
            {
                "name": "b_ub",
                "description": "A 1D tensor representing the limits for the inequality constraints."
            },
            {
                "name": "A_eq",
                "description": "A 2D tensor representing the coefficients of linear equality constraints."
            },
            {
                "name": "b_eq",
                "description": "A 1D tensor representing the targets for the equality constraints."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A structured object containing the optimal state vector."
            }
        ],
        "algorithmic_properties": [
            "linear-programming",
            "convex-optimization",
            "polyhedral-search",
            "exact-constraint-satisfaction"
        ],
        "cross_disciplinary_applications": [
            "Allocating capacity across parallel processing lines to maximize a linear throughput objective.",
            "Finding the least-cost diet that satisfies a set of nutritional requirements.",
            "Scheduling logistics deliveries to minimize total distance under capacity constraints."
        ]
    }
    <!-- /conceptual_profile -->
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
            Jacobian matrix of the model function with respect to
            parameters as a dense array_like structure.

    Returns:
        popt: Optimal values for the parameters so that the sum of the
            squared residuals of f(xdata, *popt) - ydata is minimized.
        pcov: The estimated covariance of popt.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Non-Linear Regression Model Optimizer",
        "conceptual_transform": "Optimizes the parameters of a user-defined model function to best approximate a set of observation pairs by minimizing the sum of squared residuals. It performs a non-linear least-squares refinement to match model behavior to empirical data.",
        "abstract_inputs": [
            {
                "name": "f",
                "description": "A functional mapping representing the parameterized model f(x, *parameters)."
            },
            {
                "name": "xdata",
                "description": "A 1D tensor of independent variable observations."
            },
            {
                "name": "ydata",
                "description": "A 1D tensor of dependent variable observations."
            },
            {
                "name": "p0",
                "description": "A 1D tensor representing the initial guess for the model parameters."
            }
        ],
        "abstract_outputs": [
            {
                "name": "popt",
                "description": "A 1D tensor of optimized model parameters."
            },
            {
                "name": "pcov",
                "description": "A 2D tensor representing the estimated covariance (uncertainty) of the optimized parameters."
            }
        ],
        "algorithmic_properties": [
            "non-linear-least-squares",
            "parameter-refinement",
            "stochastic-uncertainty-estimation"
        ],
        "cross_disciplinary_applications": [
            "Fitting a specific theoretical decay model to radioactive decay measurements.",
            "Extracting kinetic rate constants from time-series concentration data in chemistry.",
            "Calibrating a non-linear sensor response model using known reference standards."
        ]
    }
    <!-- /conceptual_profile -->
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
