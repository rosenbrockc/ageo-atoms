"""Graph algorithm atoms with icontract contracts and ghost witnesses.

Covers BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall.
Wraps scipy.sparse.csgraph implementations.
"""

from __future__ import annotations

from typing import Any

import icontract
import numpy as np

from ageoa.ghost.abstract import AbstractArray
from ageoa.ghost.registry import register_atom


def _witness_graph_distances(adj: AbstractArray) -> AbstractArray:
    """Ghost witness for graph distance algorithms.

    Postconditions:
        - Output is 1D with n_nodes elements.
        - Values represent distances (non-negative or inf).
    """
    if len(adj.shape) != 2:
        raise ValueError(f"Adjacency matrix must be 2D, got shape {adj.shape}")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}")
    n = adj.shape[0]
    return AbstractArray(
        shape=(n,),
        dtype="float64",
        min_val=0.0,
    )


def _witness_all_pairs_distances(adj: AbstractArray) -> AbstractArray:
    """Ghost witness for all-pairs shortest paths."""
    if len(adj.shape) != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square 2D, got shape {adj.shape}")
    n = adj.shape[0]
    return AbstractArray(
        shape=(n, n),
        dtype="float64",
        min_val=0.0,
    )


def _witness_traversal(adj: AbstractArray) -> AbstractArray:
    """Ghost witness for graph traversal (BFS/DFS).

    Returns array of visited node indices.
    """
    if len(adj.shape) != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square 2D, got shape {adj.shape}")
    n = adj.shape[0]
    return AbstractArray(
        shape=(n,),
        dtype="int64",
        is_index=True,
    )


@register_atom(witness=_witness_traversal)
@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0],),
    "BFS result must have one entry per node",
)
@icontract.ensure(
    lambda result: result.ndim == 1,
    "BFS result must be 1D",
)
def bfs(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Breadth-first search: returns BFS order of node indices.

    Args:
        adj: Square 2D adjacency matrix of shape (n, n).
        source: Starting node index. Default is 0.

    Returns:
        1D array of length n with BFS visit order; unreachable nodes are -1.
    """
    from scipy.sparse.csgraph import breadth_first_order
    from scipy.sparse import csr_matrix

    graph = csr_matrix(adj)
    order, _ = breadth_first_order(graph, source, directed=True)
    result = np.full(adj.shape[0], -1, dtype=np.intp)
    result[: len(order)] = order
    return result


@register_atom(witness=_witness_traversal)
@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0],),
    "DFS result must have one entry per node",
)
@icontract.ensure(
    lambda result: result.ndim == 1,
    "DFS result must be 1D",
)
def dfs(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Depth-first search: returns DFS order of node indices.

    Args:
        adj: Square 2D adjacency matrix of shape (n, n).
        source: Starting node index. Default is 0.

    Returns:
        1D array of length n with DFS visit order; unreachable nodes are -1.
    """
    from scipy.sparse.csgraph import depth_first_order
    from scipy.sparse import csr_matrix

    graph = csr_matrix(adj)
    order, _ = depth_first_order(graph, source, directed=True)
    result = np.full(adj.shape[0], -1, dtype=np.intp)
    result[: len(order)] = order
    return result


@register_atom(witness=_witness_graph_distances)
@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.require(
    lambda adj: np.all(adj >= 0),
    "Dijkstra requires non-negative weights",
)
@icontract.ensure(
    lambda result: np.all(result >= 0),
    "Distances must be non-negative",
)
def dijkstra(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Dijkstra's shortest paths from a single source.

    Args:
        adj: Square 2D adjacency matrix of shape (n, n) with non-negative weights.
        source: Starting node index. Default is 0.

    Returns:
        1D array of shortest distances from source to each node.
    """
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    from scipy.sparse import csr_matrix

    graph = csr_matrix(adj)
    dist = sp_dijkstra(graph, indices=source, directed=True)
    return dist


@register_atom(witness=_witness_graph_distances)
@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0],),
    "Bellman-Ford result must have one entry per node",
)
@icontract.ensure(
    lambda result: result.ndim == 1,
    "Bellman-Ford result must be 1D",
)
def bellman_ford(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Bellman-Ford shortest paths (handles negative weights).

    Args:
        adj: Square 2D adjacency matrix of shape (n, n).
        source: Starting node index. Default is 0.

    Returns:
        1D array of shortest distances from source to each node.
    """
    from scipy.sparse.csgraph import bellman_ford as sp_bf
    from scipy.sparse import csr_matrix

    graph = csr_matrix(adj)
    dist = sp_bf(graph, indices=source, directed=True)
    return dist


@register_atom(witness=_witness_all_pairs_distances)
@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0], adj.shape[1]),
    "Floyd-Warshall result must be square with same dimensions as input",
)
@icontract.ensure(
    lambda result: result.ndim == 2,
    "Floyd-Warshall result must be 2D",
)
def floyd_warshall(adj: np.ndarray) -> np.ndarray:
    """Floyd-Warshall all-pairs shortest paths.

    Args:
        adj: Square 2D adjacency matrix of shape (n, n).

    Returns:
        2D array of shape (n, n) with shortest distances between all pairs.
    """
    from scipy.sparse.csgraph import floyd_warshall as sp_fw
    from scipy.sparse import csr_matrix

    graph = csr_matrix(adj)
    dist = sp_fw(graph, directed=True)
    return dist
