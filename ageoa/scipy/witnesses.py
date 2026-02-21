"""Ghost witnesses for SciPy wrapper atoms."""

from __future__ import annotations

from typing import Any, Sequence

from ageoa.ghost.abstract import (
    AbstractArray,
    AbstractDistribution,
    AbstractScalar,
)


def _as_array_or_scalar(
    shape: tuple[int, ...],
    *,
    dtype: str = "float64",
    min_val: float | None = None,
    max_val: float | None = None,
) -> AbstractArray | AbstractScalar:
    if shape == ():
        return AbstractScalar(dtype=dtype, min_val=min_val, max_val=max_val)
    return AbstractArray(shape=shape, dtype=dtype, min_val=min_val, max_val=max_val)


def _as_array_meta(x: AbstractArray | AbstractScalar) -> AbstractArray:
    if isinstance(x, AbstractArray):
        return x
    return AbstractArray(shape=(), dtype=x.dtype, min_val=x.min_val, max_val=x.max_val)


def _shape_without_axis(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    if not shape:
        return ()
    ndim = len(shape)
    ax = axis if axis >= 0 else ndim + axis
    if ax < 0 or ax >= ndim:
        raise ValueError(f"axis {axis} out of bounds for shape {shape}")
    return shape[:ax] + shape[ax + 1 :]


def _leading_len(x: AbstractArray) -> int:
    return x.shape[0] if x.shape else 1


def witness_scipy_quad(
    func: Any,
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
) -> tuple[AbstractScalar, AbstractScalar] | tuple[AbstractScalar, AbstractScalar, AbstractScalar]:
    _ = (func, a, b, args, epsabs, epsrel, limit, points, weight, wvar, wopts, maxp1, limlst)
    integral = AbstractScalar(dtype="float64")
    abs_err = AbstractScalar(dtype="float64", min_val=0.0)
    if full_output:
        evals = AbstractScalar(dtype="int64", min_val=0.0)
        return (integral, abs_err, evals)
    return (integral, abs_err)


def witness_scipy_solve_ivp(
    fun: Any,
    t_span: tuple[float, float],
    y0: AbstractArray,
    method: str = "RK45",
    t_eval: AbstractArray | None = None,
    dense_output: bool = False,
    events: Any = None,
    vectorized: bool = False,
    args: tuple | None = None,
    **options: Any,
) -> AbstractArray:
    _ = (fun, t_span, method, dense_output, events, vectorized, args, options)
    n_state = _leading_len(y0)
    if t_eval is None:
        return AbstractArray(shape=(n_state,), dtype=y0.dtype)
    n_t = _leading_len(t_eval)
    return AbstractArray(shape=(n_state, n_t), dtype=y0.dtype)


def witness_scipy_simpson(
    y: AbstractArray,
    x: AbstractArray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> AbstractArray | AbstractScalar:
    _ = (x, dx)
    if len(y.shape) <= 1:
        return AbstractScalar(dtype="float64")
    out_shape = _shape_without_axis(y.shape, axis)
    return _as_array_or_scalar(out_shape, dtype="float64")


def witness_scipy_linalg_solve(
    a: AbstractArray,
    b: AbstractArray,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str = "gen",
) -> AbstractArray:
    _ = (lower, overwrite_a, overwrite_b, check_finite, assume_a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    if not b.shape or b.shape[0] != a.shape[0]:
        raise ValueError(f"Incompatible shapes for solve: a={a.shape}, b={b.shape}")
    return AbstractArray(shape=b.shape, dtype=a.dtype)


def witness_scipy_linalg_inv(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> AbstractArray:
    _ = (overwrite_a, check_finite)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    return AbstractArray(shape=a.shape, dtype=a.dtype)


def witness_scipy_linalg_det(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> AbstractArray | AbstractScalar:
    _ = (overwrite_a, check_finite)
    if len(a.shape) < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"a must be at least 2D with square trailing dims, got {a.shape}")
    out_shape = a.shape[:-2]
    return _as_array_or_scalar(out_shape, dtype="float64")


def witness_scipy_lu_factor(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[AbstractArray, AbstractArray]:
    _ = (overwrite_a, check_finite)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    n = a.shape[0]
    return (
        AbstractArray(shape=a.shape, dtype=a.dtype),
        AbstractArray(shape=(n,), dtype="int64", min_val=0.0, max_val=float(max(n - 1, 0))),
    )


def witness_scipy_lu_solve(
    lu_and_piv: tuple[AbstractArray, AbstractArray],
    b: AbstractArray,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> AbstractArray:
    _ = (trans, overwrite_b, check_finite)
    lu, piv = lu_and_piv
    if len(lu.shape) != 2 or lu.shape[0] != lu.shape[1]:
        raise ValueError(f"lu must be square 2D, got {lu.shape}")
    if piv.shape != (lu.shape[0],):
        raise ValueError(f"piv shape must be {(lu.shape[0],)}, got {piv.shape}")
    if not b.shape or b.shape[0] != lu.shape[0]:
        raise ValueError(f"Incompatible shapes for lu_solve: lu={lu.shape}, b={b.shape}")
    return AbstractArray(shape=b.shape, dtype=lu.dtype)


def witness_scipy_minimize(
    fun: Any,
    x0: AbstractArray,
    args: tuple = (),
    method: str | None = None,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Sequence | None = None,
    constraints: Any = (),
    tol: float | None = None,
    callback: Any = None,
    options: dict | None = None,
) -> AbstractArray:
    _ = (fun, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    return AbstractArray(shape=x0.shape, dtype="float64")


def witness_scipy_root(
    fun: Any,
    x0: AbstractArray,
    args: tuple = (),
    method: str = "hybr",
    jac: Any = None,
    tol: float | None = None,
    callback: Any = None,
    options: dict | None = None,
) -> AbstractArray:
    _ = (fun, args, method, jac, tol, callback, options)
    return AbstractArray(shape=x0.shape, dtype="float64")


def witness_scipy_linprog(
    c: AbstractArray,
    A_ub: AbstractArray | None = None,
    b_ub: AbstractArray | None = None,
    A_eq: AbstractArray | None = None,
    b_eq: AbstractArray | None = None,
    bounds: Sequence | None = None,
    method: str = "highs",
    callback: Any = None,
    options: dict | None = None,
    x0: AbstractArray | None = None,
) -> AbstractArray:
    _ = (A_ub, b_ub, A_eq, b_eq, bounds, method, callback, options, x0)
    n_vars = _leading_len(c)
    return AbstractArray(shape=(n_vars,), dtype="float64")


def witness_scipy_curve_fit(
    f: Any,
    xdata: AbstractArray,
    ydata: AbstractArray,
    p0: AbstractArray | None = None,
    sigma: AbstractArray | None = None,
    absolute_sigma: bool = False,
    check_finite: bool | None = None,
    bounds: Sequence | None = (-float("inf"), float("inf")),
    method: str | None = None,
    jac: Any = None,
    **kwargs: Any,
) -> tuple[AbstractArray, AbstractArray]:
    _ = (f, sigma, absolute_sigma, check_finite, bounds, method, jac, kwargs)
    if _leading_len(xdata) != _leading_len(ydata):
        raise ValueError(f"xdata and ydata must have same length, got {xdata.shape} and {ydata.shape}")
    if p0 is None:
        n_params = 1
    else:
        n_params = _leading_len(p0)
    return (
        AbstractArray(shape=(n_params,), dtype="float64"),
        AbstractArray(shape=(n_params, n_params), dtype="float64"),
    )


def witness_scipy_describe(
    a: AbstractArray,
    axis: int | None = 0,
    ddof: int = 1,
    bias: bool = True,
    nan_policy: str = "propagate",
) -> tuple[AbstractScalar, tuple[AbstractScalar, AbstractScalar], AbstractScalar, AbstractScalar, AbstractScalar, AbstractScalar]:
    _ = (a, axis, ddof, bias, nan_policy)
    return (
        AbstractScalar(dtype="int64", min_val=1.0),
        (AbstractScalar(dtype="float64"), AbstractScalar(dtype="float64")),
        AbstractScalar(dtype="float64"),
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="float64"),
        AbstractScalar(dtype="float64"),
    )


def witness_scipy_ttest_ind(
    a: AbstractArray,
    b: AbstractArray,
    axis: int = 0,
    equal_var: bool = True,
    nan_policy: str = "propagate",
    permutations: float | None = None,
    random_state: int | None = None,
    alternative: str = "two-sided",
    trim: float = 0,
) -> tuple[AbstractScalar, AbstractScalar]:
    _ = (axis, equal_var, nan_policy, permutations, random_state, alternative, trim)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have matching shapes, got {a.shape} and {b.shape}")
    return (
        AbstractScalar(dtype="float64"),
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
    )


def witness_scipy_pearsonr(
    x: AbstractArray,
    y: AbstractArray,
) -> tuple[AbstractScalar, AbstractScalar]:
    if _leading_len(x) != _leading_len(y):
        raise ValueError(f"x and y must have same length, got {x.shape} and {y.shape}")
    return (
        AbstractScalar(dtype="float64", min_val=-1.0, max_val=1.0),
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
    )


def witness_scipy_spearmanr(
    a: AbstractArray,
    b: AbstractArray | None = None,
    axis: int | None = 0,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
) -> tuple[AbstractScalar, AbstractScalar]:
    _ = (axis, nan_policy, alternative)
    if b is not None and _leading_len(a) != _leading_len(b):
        raise ValueError(f"a and b must have same length, got {a.shape} and {b.shape}")
    return (
        AbstractScalar(dtype="float64", min_val=-1.0, max_val=1.0),
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
    )


def witness_scipy_norm(
    loc: float = 0.0,
    scale: float = 1.0,
) -> AbstractDistribution:
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return AbstractDistribution(
        family="normal",
        event_shape=(),
        support_lower=None,
        support_upper=None,
        is_discrete=False,
    )
