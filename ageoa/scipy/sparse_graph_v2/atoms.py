from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
from typing import Any
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import (
    witness_singlesourceshortestpath,
    witness_allpairsshortestpath,
    witness_minimumspanningtree,
)


@register_atom(witness_singlesourceshortestpath)
@icontract.require(lambda limit: isinstance(limit, (float, int, np.number)), "limit must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "SingleSourceShortestPath all outputs must not be None")
def singlesourceshortestpath(csgraph: Any, directed: bool, indices: Any, return_predecessors: bool, unweighted: bool, limit: float, min_only: bool) -> Any:
    """Computes shortest-path distances from one or more source nodes to all reachable nodes. Dijkstra's greedy algorithm is used for non-negative weights; Bellman-Ford's edge-relaxation handles negative weights. Both share the same intent - locate optimal routes from a set of origins - and return a distance matrix plus an optional predecessor map.

    Args:
        csgraph: valid scipy sparse graph; Dijkstra requires non-negative weights
        directed: treat graph as directed if True
        indices: source node indices; None means all nodes
        return_predecessors: Input data.
        unweighted: treat all edges as weight-1 if True
        limit: Dijkstra only; paths longer than limit are not explored
        min_only: Dijkstra only; return scalar minimum distance per source instead of full row

    Returns:
        dist_matrix: inf where no path exists; K = len(indices) or N when indices=None
        predecessors: emitted only when return_predecessors=True; -9999 for unreachable nodes
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_allpairsshortestpath)
@icontract.require(lambda csgraph: csgraph is not None, "csgraph cannot be None")
@icontract.require(lambda directed: directed is not None, "directed cannot be None")
@icontract.require(lambda return_predecessors: return_predecessors is not None, "return_predecessors cannot be None")
@icontract.require(lambda unweighted: unweighted is not None, "unweighted cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "AllPairsShortestPath all outputs must not be None")
def allpairsshortestpath(csgraph: Any, directed: bool, return_predecessors: bool, unweighted: bool) -> Any:
    """Computes the full N×N shortest-path distance matrix for every source–destination pair simultaneously using the Floyd-Warshall dynamic-programming recurrence. Suitable for dense reachability queries where all pairwise distances are required at once.

    Args:
        csgraph: valid scipy sparse graph
        directed: Input data.
        return_predecessors: Input data.
        unweighted: treat all edges as weight-1 if True

    Returns:
        dist_matrix: inf where no path exists; diagonal is 0
        predecessors: emitted only when return_predecessors=True
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_minimumspanningtree)
@icontract.require(lambda csgraph: csgraph is not None, "csgraph cannot be None")
@icontract.require(lambda overwrite: overwrite is not None, "overwrite cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "MinimumSpanningTree output must not be None")
def minimumspanningtree(csgraph: Any, overwrite: bool) -> Any:
    """Extracts the Minimum Spanning Tree (MST) of a sparse undirected weighted graph. Returns a new sparse matrix containing only the tree edges that connect all nodes with minimum total weight.

    Args:
        csgraph: symmetric/undirected; non-negative edge weights
        overwrite: if True the input matrix may be clobbered; avoids an internal copy

    Returns:
        upper-triangular; contains exactly N-1 edges for a connected graph; edge weights preserved from input
    """
    raise NotImplementedError("Wire to original implementation")
