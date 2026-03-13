from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_cubicsplinefit(x: AbstractArray, y: AbstractArray, axis: AbstractArray, bc_type: AbstractArray, extrapolate: AbstractArray) -> AbstractArray:
    """Shape-and-type check for cubic spline fit. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",)
    
    return result

def witness_rbfinterpolatorfit(y: AbstractArray, d: AbstractArray, neighbors: AbstractArray, smoothing: AbstractArray, kernel: AbstractArray, epsilon: AbstractArray, degree: AbstractArray) -> AbstractArray:
    """Shape-and-type check for rbf interpolator fit. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=y.shape,
        dtype="float64",)
    
    return result