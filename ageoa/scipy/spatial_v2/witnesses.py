from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_voronoitessellation(points: AbstractArray, incremental: AbstractArray, qhull_options: AbstractArray) -> AbstractArray:
    """Ghost witness for VoronoiTessellation."""
    result = AbstractArray(
        shape=points.shape,
        dtype="float64",)
    
    return result

def witness_delaunaytriangulation(points: AbstractArray, incremental: AbstractArray, qhull_options: AbstractArray) -> AbstractArray:
    """Ghost witness for DelaunayTriangulation."""
    result = AbstractArray(
        shape=points.shape,
        dtype="float64",)
    
    return result