from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_cubicsplinefit(x: AbstractArray, y: AbstractArray, axis: AbstractArray, bc_type: AbstractArray, extrapolate: AbstractArray) -> AbstractArray:
    """Ghost witness for CubicSplineFit."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",)
    
    return result

def witness_rbfinterpolatorfit(y: AbstractArray, d: AbstractArray, neighbors: AbstractArray, smoothing: AbstractArray, kernel: AbstractArray, epsilon: AbstractArray, degree: AbstractArray) -> AbstractArray:
    """Ghost witness for RBFInterpolatorFit."""
    result = AbstractArray(
        shape=y.shape,
        dtype="float64",)
    
    return result