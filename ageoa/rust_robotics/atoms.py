"""Auto-generated verified atom wrapper."""

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_dijkstra_path_planning, witness_n_joint_arm_solver



@register_atom(witness_n_joint_arm_solver)
@icontract.require(lambda data: np.isfinite(data).all(), "data must contain only finite values")
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
@icontract.require(lambda data: data.ndim >= 1, "data must have at least one dimension")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def n_joint_arm_solver(data: np.ndarray) -> np.ndarray:
    """Solves custom kinematics and dynamics for an N-joint system.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    angles = data.ravel()
    cumulative = np.cumsum(angles)
    x = np.sum(np.cos(cumulative))
    y = np.sum(np.sin(cumulative))
    return np.array([x, y])

@register_atom(witness_dijkstra_path_planning)
@icontract.require(lambda data: np.isfinite(data).all(), "data must contain only finite values")
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
@icontract.require(lambda data: data.ndim >= 1, "data must have at least one dimension")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def dijkstra_path_planning(data: np.ndarray) -> np.ndarray:
    """Computes the shortest path on a weighted graph from a single source node.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    adj = np.array(data, dtype=float)
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    dist[0] = 0.0
    visited = np.zeros(n, dtype=bool)
    for _ in range(n):
        # pick unvisited node with smallest distance
        unvisited_dist = np.where(visited, np.inf, dist)
        u = int(np.argmin(unvisited_dist))
        if dist[u] == np.inf:
            break
        visited[u] = True
        for v in range(n):
            if adj[u, v] > 0 and not visited[v]:
                alt = dist[u] + adj[u, v]
                if alt < dist[v]:
                    dist[v] = alt
    return dist
