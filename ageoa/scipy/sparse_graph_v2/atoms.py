from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""
import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import (
    witness_singlesourceshortestpath,
    witness_allpairsshortestpath,
    witness_minimumspanningtree,
)


@register_atom(witness_singlesourceshortestpath)
@icontract.require(lambda limit: isinstance(limit, (float, int, np.number)), "limit must be numeric")
@icontract.ensure(lambda result: result is not None, "SingleSourceShortestPath output must not be None")
def singlesourceshortestpath(
    csgraph: np.ndarray,
    directed: bool = True,
    indices: np.ndarray | int | None = None,
    return_predecessors: bool = False,
    unweighted: bool = False,
    limit: float = np.inf,
    min_only: bool = False,
) -> tuple[np.ndarray, ...] | np.ndarray:
    """Compute shortest-path distances from one or more source nodes with SciPy's ``dijkstra``.

    Args:
        csgraph: Sparse graph accepted by ``scipy.sparse.csgraph.dijkstra``.
        directed: Treat edges as directed when ``True``.
        indices: Source node index or indices; ``None`` means all nodes.
        return_predecessors: Return the predecessor matrix alongside distances.
        unweighted: Treat every nonzero edge as weight ``1``.
        limit: Ignore paths whose cumulative cost exceeds this bound.
        min_only: Return only the minimum-distance vector when supported.

    Returns:
        Distance matrix or vector, plus predecessors when requested.
    """
    from scipy.sparse.csgraph import dijkstra
    return dijkstra(csgraph, directed=directed, indices=indices, return_predecessors=return_predecessors, unweighted=unweighted, limit=limit, min_only=min_only)

@register_atom(witness_allpairsshortestpath)
@icontract.require(lambda csgraph: csgraph is not None, "csgraph cannot be None")
@icontract.ensure(lambda result: result is not None, "AllPairsShortestPath output must not be None")
def allpairsshortestpath(
    csgraph: np.ndarray,
    directed: bool = True,
    return_predecessors: bool = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[np.ndarray, ...] | np.ndarray:
    """Compute all-pairs shortest paths with SciPy's ``floyd_warshall``.

    Args:
        csgraph: Sparse graph accepted by ``scipy.sparse.csgraph.floyd_warshall``.
        directed: Treat edges as directed when ``True``.
        return_predecessors: Return the predecessor matrix alongside distances.
        unweighted: Treat every nonzero edge as weight ``1``.
        overwrite: Allow SciPy to overwrite the working copy in place.

    Returns:
        Full distance matrix, plus predecessors when requested.
    """
    from scipy.sparse.csgraph import floyd_warshall
    return floyd_warshall(
        csgraph,
        directed=directed,
        return_predecessors=return_predecessors,
        unweighted=unweighted,
        overwrite=overwrite,
    )

@register_atom(witness_minimumspanningtree)
@icontract.require(lambda csgraph: csgraph is not None, "csgraph cannot be None")
@icontract.ensure(lambda result: result is not None, "MinimumSpanningTree output must not be None")
def minimumspanningtree(csgraph: np.ndarray, overwrite: bool = False) -> np.ndarray:
    """Extract the minimum spanning tree of a sparse weighted graph.

    Args:
        csgraph: Sparse weighted graph accepted by ``scipy.sparse.csgraph.minimum_spanning_tree``.
        overwrite: Allow SciPy to overwrite the working copy in place.

    Returns:
        Sparse matrix containing the spanning-tree edges.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree as _mst
    return _mst(csgraph, overwrite=overwrite)
