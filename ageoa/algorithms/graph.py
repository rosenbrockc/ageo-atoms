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
def bfs(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Breadth-first search: returns BFS order of node indices.

    <!-- conceptual_profile
    {
        "abstract_name": "Radial Connectivity Explorer",
        "conceptual_transform": "Traverses a topological relational structure by visiting all immediate neighbors of a seed element before moving to the next level of connectivity. It generates a sequence representing the minimal step-distance order from the origin.",
        "abstract_inputs": [
            {
                "name": "adj",
                "description": "A square 2D tensor representing the connectivity matrix (adjacency) of a relational system."
            },
            {
                "name": "source",
                "description": "An integer identifier for the seed element from which to begin exploration."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of identifiers representing the order in which elements were reached."
            }
        ],
        "algorithmic_properties": [
            "radial-traversal",
            "level-order",
            "deterministic"
        ],
        "cross_disciplinary_applications": [
            "Modeling the spread of an infection through a social contact network.",
            "Finding the shortest path in an unweighted maze or routing grid.",
            "Resolving dependencies in a software build system to determine a safe compilation order."
        ]
    }
    /conceptual_profile -->
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
def dfs(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Depth-first search: returns DFS order of node indices.

    <!-- conceptual_profile
    {
        "abstract_name": "Branching Path Recursion Explorer",
        "conceptual_transform": "Traverses a topological relational structure by following a single path to its maximum depth before backtracking to explore alternate branches. It captures the deep hierarchical structure of a connectivity map.",
        "abstract_inputs": [
            {
                "name": "adj",
                "description": "A square 2D tensor representing the connectivity matrix of a relational system."
            },
            {
                "name": "source",
                "description": "An integer identifier for the seed element from which to begin exploration."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of identifiers representing the depth-first visit order."
            }
        ],
        "algorithmic_properties": [
            "recursive-traversal",
            "backtracking",
            "depth-priority"
        ],
        "cross_disciplinary_applications": [
            "Detecting cycles in a complex chemical reaction network.",
            "Searching for a solution in a game tree with high branching factors.",
            "Identifying strongly connected components in a functional dependency graph."
        ]
    }
    /conceptual_profile -->
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

    <!-- conceptual_profile
    {
        "abstract_name": "Single-Source Optimal Path Transformer",
        "conceptual_transform": "Computes the minimum cumulative cost from a single seed element to all other reachable elements within a non-negative cost field. It effectively maps a connectivity structure to a distance field.",
        "abstract_inputs": [
            {
                "name": "adj",
                "description": "A square 2D tensor where values represent the non-negative cost of transition between elements."
            },
            {
                "name": "source",
                "description": "An integer identifier for the seed element."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of floats representing the minimum cumulative cost to reach each element."
            }
        ],
        "algorithmic_properties": [
            "greedy-optimization",
            "non-negative-constrained",
            "path-minimization"
        ],
        "cross_disciplinary_applications": [
            "Calculating the lowest-latency route for data packets in a wide-area network.",
            "Optimizing logistics delivery routes based on fuel consumption or time costs.",
            "Computing minimum-cost traversal paths in a weighted spatial grid."
        ]
    }
    /conceptual_profile -->
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
def bellman_ford(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Bellman-Ford shortest paths (handles negative weights).

    <!-- conceptual_profile
    {
        "abstract_name": "Dynamic Relational Cost Solver",
        "conceptual_transform": "Computes the minimum cumulative cost from a seed element to all others, capable of handling negative transition costs. It iteratively relaxes constraints until a global minimum is reached or a self-amplifying cycle is detected.",
        "abstract_inputs": [
            {
                "name": "adj",
                "description": "A square 2D tensor representing transition costs (can be negative)."
            },
            {
                "name": "source",
                "description": "An integer identifier for the seed element."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of floats representing the minimum cumulative cost to each element."
            }
        ],
        "algorithmic_properties": [
            "iterative-relaxation",
            "negative-cost-compatible",
            "cycle-detecting"
        ],
        "cross_disciplinary_applications": [
            "Identifying arbitrage opportunities in a currency exchange network with transaction costs.",
            "Modeling state transitions in systems with potential energy wells and gains.",
            "Solving optimal control problems where 'negative' costs represent energy recovery."
        ]
    }
    /conceptual_profile -->
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
def floyd_warshall(adj: np.ndarray) -> np.ndarray:
    """Floyd-Warshall all-pairs shortest paths.

    <!-- conceptual_profile
    {
        "abstract_name": "Global Connectivity Cost Matrix Generator",
        "conceptual_transform": "Computes the shortest path distance between every pair of elements in a relational system simultaneously. It transforms a local connectivity matrix into a global all-to-all proximity matrix.",
        "abstract_inputs": [
            {
                "name": "adj",
                "description": "A square 2D tensor representing local transition costs between elements."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A square 2D tensor representing the global minimum cost between every pair of elements."
            }
        ],
        "algorithmic_properties": [
            "dynamic-programming",
            "all-pairs",
            "global-state-resolution"
        ],
        "cross_disciplinary_applications": [
            "Computing the diameter and average path length of a biological protein interaction network.",
            "Precomputing a distance lookup table for all city pairs in a transportation system.",
            "Analyzing the robustness of a communications network by identifying critical bottlenecks."
        ]
    }
    /conceptual_profile -->
    """
    from scipy.sparse.csgraph import floyd_warshall as sp_fw
    from scipy.sparse import csr_matrix

    graph = csr_matrix(adj)
    dist = sp_fw(graph, directed=True)
    return dist
