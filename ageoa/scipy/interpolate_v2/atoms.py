from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import witness_cubicsplinefit, witness_rbfinterpolatorfit

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_cubicsplinefit)
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda y: y is not None, "y cannot be None")
@icontract.require(lambda axis: axis is not None, "axis cannot be None")
@icontract.require(lambda bc_type: bc_type is not None, "bc_type cannot be None")
@icontract.require(lambda extrapolate: extrapolate is not None, "extrapolate cannot be None")
@icontract.ensure(lambda result: result is not None, "CubicSplineFit output must not be None")
def cubicsplinefit(x: np.ndarray, y: np.ndarray, axis: int = 0, bc_type: str | tuple | None = None, extrapolate: bool | str | None = None) -> object:  # type: ignore[type-arg]
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
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_rbfinterpolatorfit)
@icontract.require(lambda y: isinstance(y, (float, int, np.number)), "y must be numeric")
@icontract.require(lambda smoothing: isinstance(smoothing, (float, int, np.number)), "smoothing must be numeric")
@icontract.require(lambda epsilon: isinstance(epsilon, (float, int, np.number)), "epsilon must be numeric")
@icontract.ensure(lambda result: result is not None, "RBFInterpolatorFit output must not be None")
def rbfinterpolatorfit(y: np.ndarray, d: np.ndarray, neighbors: int | None = None, smoothing: float | np.ndarray | None = None, kernel: str | None = None, epsilon: float | None = None, degree: int | None = None) -> object:  # type: ignore[type-arg]
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
    raise NotImplementedError("Wire to original implementation")
