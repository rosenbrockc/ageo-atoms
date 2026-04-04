from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_cubicsplinefit(x: AbstractArray, y: AbstractArray, axis: AbstractScalar, bc_type: AbstractScalar, extrapolate: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for cubic spline fit. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=("callable",),
        dtype="object",)
    
    return result

def witness_rbfinterpolatorfit(y: AbstractArray, d: AbstractArray, neighbors: AbstractScalar, smoothing: AbstractArray, kernel: AbstractScalar, epsilon: AbstractScalar, degree: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for rbf interpolator fit. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=("callable",),
        dtype="object",)
    
    return result
