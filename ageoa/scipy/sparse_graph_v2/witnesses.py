from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_singlesourceshortestpath(csgraph: AbstractArray, directed: AbstractArray, indices: AbstractArray, return_predecessors: AbstractArray, unweighted: AbstractArray, limit: AbstractArray, min_only: AbstractArray) -> AbstractArray:
    """Shape-and-type check for single source shortest path. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=csgraph.shape,
        dtype="float64",)
    
    return result

def witness_allpairsshortestpath(csgraph: AbstractArray, directed: AbstractArray, return_predecessors: AbstractArray, unweighted: AbstractArray) -> AbstractArray:
    """Shape-and-type check for all pairs shortest path. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=csgraph.shape,
        dtype="float64",)
    
    return result

def witness_minimumspanningtree(csgraph: AbstractArray, overwrite: AbstractArray) -> AbstractArray:
    """Shape-and-type check for minimum spanning tree. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=csgraph.shape,
        dtype="float64",)
    
    return result