from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from typing import TYPE_CHECKING

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import witness_cubicsplinefit, witness_rbfinterpolatorfit

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline, RBFInterpolator

# Witness functions should be imported from the generated witnesses module


def _is_array(value: object, *, ndim: int | None = None) -> bool:
    """Return whether ``value`` is a NumPy array with an optional rank constraint."""
    if not isinstance(value, np.ndarray):
        return False
    if ndim is not None and value.ndim != ndim:
        return False
    return np.all(np.isfinite(value))


def _is_axis_index(value: object) -> bool:
    """Return whether ``value`` is an integer axis index."""
    return isinstance(value, (int, np.integer))


def _is_boundary_condition(value: object) -> bool:
    """Return whether ``value`` is a valid boundary-condition shape for ``CubicSpline``."""
    return value is None or isinstance(value, (str, tuple))


def _is_extrapolation_mode(value: object) -> bool:
    """Return whether ``value`` is an accepted extrapolation mode for ``CubicSpline``."""
    return value is None or isinstance(value, (bool, str))


def _is_optional_int(value: object) -> bool:
    """Return whether ``value`` is ``None`` or an integer."""
    return value is None or isinstance(value, (int, np.integer))


def _is_optional_smoothing(value: object) -> bool:
    """Return whether ``value`` is a valid smoothing parameter for ``RBFInterpolator``."""
    return value is None or isinstance(value, (float, int, np.number, np.ndarray))


def _is_optional_str(value: object) -> bool:
    """Return whether ``value`` is ``None`` or a string."""
    return value is None or isinstance(value, str)


def _is_optional_float(value: object) -> bool:
    """Return whether ``value`` is ``None`` or a numeric scalar."""
    return value is None or isinstance(value, (float, int, np.number))

@register_atom(witness_cubicsplinefit)
@icontract.require(lambda x: _is_array(x, ndim=1), "x must be a finite 1D ndarray")
@icontract.require(lambda y: _is_array(y), "y must be a finite ndarray")
@icontract.require(lambda axis: _is_axis_index(axis), "axis must be an integer axis index")
@icontract.require(lambda bc_type: _is_boundary_condition(bc_type), "bc_type must be a string or a tuple boundary specification")
@icontract.require(lambda extrapolate: _is_extrapolation_mode(extrapolate), "extrapolate must be None, bool, or string")
@icontract.ensure(lambda result: result is not None, "CubicSplineFit output must not be None")
def cubicsplinefit(x: np.ndarray, y: np.ndarray, axis: int = 0, bc_type: str | tuple = "not-a-knot", extrapolate: bool | str | None = None) -> CubicSpline:
    """Constructs a piecewise cubic polynomial interpolator through 1-D data points, exposing boundary-condition and extrapolation policy as configuration knobs. Returns a callable CubicSpline object that evaluates (and optionally differentiates) the interpolant at arbitrary query points.

    Args:
        x: monotonically increasing; length >= 2
        y: size along `axis` must equal len(x)
        axis: valid axis index of y; defaults to 0
        bc_type: one of 'not-a-knot', 'periodic', 'clamped', 'natural', or explicit first/second derivative pairs
        extrapolate: controls whether the spline is evaluated outside the data range

    Returns:
        supports __call__(x_new), .derivative(), .antiderivative()
    """
    from scipy.interpolate import CubicSpline
    return CubicSpline(x, y, axis=axis, bc_type=bc_type, extrapolate=extrapolate)

@register_atom(witness_rbfinterpolatorfit)
@icontract.require(lambda y: _is_array(y, ndim=2), "y must be a finite 2D ndarray of coordinates")
@icontract.require(lambda d: _is_array(d), "d must be a finite ndarray of values")
@icontract.require(lambda neighbors: _is_optional_int(neighbors), "neighbors must be None or an integer")
@icontract.require(lambda smoothing: _is_optional_smoothing(smoothing), "smoothing must be a numeric scalar or an ndarray")
@icontract.require(lambda kernel: _is_optional_str(kernel), "kernel must be a string")
@icontract.require(lambda epsilon: _is_optional_float(epsilon), "epsilon must be None or a numeric scalar")
@icontract.require(lambda degree: _is_optional_int(degree), "degree must be None or an integer")
@icontract.ensure(lambda result: result is not None, "RBFInterpolatorFit output must not be None")
def rbfinterpolatorfit(y: np.ndarray, d: np.ndarray, neighbors: int | None = None, smoothing: float | np.ndarray = 0.0, kernel: str = "thin_plate_spline", epsilon: float | None = None, degree: int | None = None) -> RBFInterpolator:
    """Constructs a Radial Basis Function interpolator over scattered N-dimensional data. Supports local approximation via k-nearest-neighbor subsets, smoothing regularisation, a choice of radial kernels, a kernel shape parameter, and an optional polynomial augmentation degree. Returns a callable RBFInterpolator object.

    Args:
        y: all values finite; n_points >= 1
        d: all values finite; leading dimension must equal n_points
        neighbors: number of nearest neighbours used per query point; None means use all n_points
        smoothing: >= 0; 0 produces exact interpolation
        kernel: one of 'linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic', 'gaussian'
        epsilon: > 0; shape/scale parameter required by 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic', 'gaussian'
        degree: degree of supplementary polynomial; -1 disables polynomial augmentation; minimum degree enforced per kernel

    Returns:
        supports __call__(x_new) where x_new is shape (m_points, n_dims)
    """
    from scipy.interpolate import RBFInterpolator
    return RBFInterpolator(y, d, neighbors=neighbors, smoothing=smoothing, kernel=kernel, epsilon=epsilon, degree=degree)
