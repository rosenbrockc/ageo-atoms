import numpy as np
import scipy.integrate
import icontract
from typing import Union, Any, Callable, Sequence, Tuple, Optional

# Types
ArrayLike = Union[np.ndarray, list, tuple]

@icontract.require(lambda func: func is not None, "Function must not be None")
@icontract.ensure(lambda result: len(result) >= 2, "Result must be a tuple containing at least (y, abserr)")
def quad(
    func: Callable,
    a: float,
    b: float,
    args: tuple = (),
    full_output: int = 0,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 50,
    points: Sequence | None = None,
    weight: str | None = None,
    wvar: float | complex | None = None,
    wopts: tuple | None = None,
    maxp1: int = 50,
    limlst: int = 50,
) -> Tuple[float, float, Any]:
    """Compute a definite integral.

    Args:
        func: A Python function or method to integrate.
        a: Lower limit of integration.
        b: Upper limit of integration.
        args: Extra arguments to pass to func.
        full_output: Non-zero to return a dictionary of integration information.
        epsabs: Absolute error tolerance.
        epsrel: Relative error tolerance.
        limit: An upper bound on the number of subintervals used in the
            adaptive algorithm.
        points: A sequence of additional points of interest in the
            integration interval.
        weight: Type of weighting function.
        wvar: Parameters for weighting function.
        wopts: Optional weighting parameters.
        maxp1: Maximum number of Chebyshev moments.
        limlst: Upper bound on the number of cycles for oscillation.

    Returns:
        y: The integral of func from a to b.
        abserr: An estimate of the absolute error in the result.
    
    """
    return scipy.integrate.quad(
        func,
        a,
        b,
        args=args,
        full_output=full_output,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        points=points,
        weight=weight,
        wvar=wvar,
        wopts=wopts,
        maxp1=maxp1,
        limlst=limlst,
    )

@icontract.require(lambda fun, t_span, y0: fun is not None and t_span is not None and y0 is not None, "ODE function, time span, and initial condition must not be None")
@icontract.ensure(lambda result: result is not None, "ODE solution result must not be None")
def solve_ivp(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: ArrayLike,
    method: str = "RK45",
    t_eval: ArrayLike | None = None,
    dense_output: bool = False,
    events: Callable | Sequence[Callable] | None = None,
    vectorized: bool = False,
    args: tuple | None = None,
    **options: Any,
) -> Any:
    """Solve an initial value problem for a system of ODEs.

    Args:
        fun: Right-hand side of the system.
        t_span: Interval of integration (t0, tf).
        y0: Initial state.
        method: Integration method to use.
        t_eval: Times at which to store the computed solution.
        dense_output: Whether to compute a continuous solution.
        events: Events to track.
        vectorized: Whether fun is implemented in a vectorized fashion.
        args: Extra arguments to pass to fun.
        **options: Options passed to a chosen solver.

    Returns:
        OdeResult object with solution information.
    
    """
    return scipy.integrate.solve_ivp(
        fun,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options,
    )

@icontract.require(lambda y: len(np.asarray(y)) > 0, "Input y must not be empty")
def simpson(
    y: ArrayLike,
    x: ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    """Integrate y(x) using samples along the given axis and the composite
    Simpson's rule.

    Args:
        y: Array of objects to be integrated.
        x: If given, the points at which y is sampled.
        dx: Spacing of integration points when x is None.
        axis: Axis along which to integrate.

    Returns:
        Definite integral of y(x).
    
    """
    return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
