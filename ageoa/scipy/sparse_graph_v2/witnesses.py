from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar


def witness_singlesourceshortestpath(
    csgraph: AbstractArray,
    directed: AbstractScalar,
    indices: AbstractArray | AbstractScalar | None,
    return_predecessors: AbstractScalar,
    unweighted: AbstractScalar,
    limit: AbstractScalar,
    min_only: AbstractScalar,
) -> AbstractArray:
    """Return distance-array metadata for the Dijkstra wrapper."""
    return AbstractArray(shape=csgraph.shape, dtype="float64")

def witness_allpairsshortestpath(
    csgraph: AbstractArray,
    directed: AbstractScalar,
    return_predecessors: AbstractScalar,
    unweighted: AbstractScalar,
    overwrite: AbstractScalar,
) -> AbstractArray:
    """Return distance-matrix metadata for the Floyd-Warshall wrapper."""
    return AbstractArray(shape=csgraph.shape, dtype="float64")

def witness_minimumspanningtree(csgraph: AbstractArray, overwrite: AbstractScalar) -> AbstractArray:
    """Return sparse-matrix-shaped metadata for the MST wrapper."""
    return AbstractArray(shape=csgraph.shape, dtype="float64")
