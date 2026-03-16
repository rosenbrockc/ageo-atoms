from __future__ import annotations
"""Auto-generated verified atom wrapper."""

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_greedy_lattice_mapping, witness_quantum_mwis_solver

@register_atom(witness_quantum_mwis_solver)
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
@icontract.require(lambda data: data.ndim >= 1, "data must have at least one dimension")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def quantum_mwis_solver(data: np.ndarray) -> np.ndarray:
    """Solves the Maximum Weight Independent Set problem on a graph using quantum heuristics.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    from scipy.spatial.distance import pdist, squareform

    # If 2D, interpret as adjacency + weights: use simulated annealing on QUBO
    if data.ndim >= 2:
        n = data.shape[0]
        # Build QUBO: diagonal = -weights (node degree as proxy), off-diag = penalty
        adj = (data != 0).astype(float)
        np.fill_diagonal(adj, 0)
        weights = np.sum(np.abs(data), axis=1)
        penalty = 2.0 * np.max(weights) + 1.0
        Q = -np.diag(weights) + penalty * adj

        # Simulated annealing on binary vector
        rng = np.random.RandomState(42)
        x = np.zeros(n, dtype=float)
        energy = 0.0
        T = 1.0
        for step in range(1000):
            T = max(1.0 / (1 + step), 1e-6)
            i = rng.randint(n)
            x_new = x.copy()
            x_new[i] = 1.0 - x_new[i]
            e_new = x_new @ Q @ x_new
            delta = e_new - energy
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                x = x_new
                energy = e_new
        return x
    else:
        # 1D: return binary indicator of above-median values
        median = np.median(data)
        return (data >= median).astype(float)

@register_atom(witness_greedy_lattice_mapping)
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty along first axis")
@icontract.require(lambda data: data.ndim >= 2, "data must have at least two dimensions for lattice mapping")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def greedy_lattice_mapping(data: np.ndarray) -> np.ndarray:
    """Maps abstract interaction graphs onto physical 2D lattices under hardware constraints.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    from scipy.spatial.distance import cdist

    n = data.shape[0]
    # Build adjacency from input (treat as interaction matrix)
    adj = (data != 0).astype(float)
    np.fill_diagonal(adj, 0)

    # Compute node degrees for greedy ordering
    degrees = np.sum(adj, axis=1)
    sorted_nodes = np.argsort(-degrees)

    # Generate 2D lattice positions: ceil(sqrt(n)) x ceil(sqrt(n))
    side = int(np.ceil(np.sqrt(n)))
    lattice_positions = np.array([[i, j] for i in range(side) for j in range(side)])

    # Greedy assignment: assign highest-degree nodes to most central lattice positions
    center = np.array([side / 2.0, side / 2.0])
    lattice_dists = np.linalg.norm(lattice_positions - center, axis=1)
    sorted_lattice = np.argsort(lattice_dists)

    # Build mapping: node index -> lattice index
    mapping = np.full(n, -1, dtype=int)
    for rank, node_idx in enumerate(sorted_nodes):
        if rank < len(sorted_lattice):
            mapping[node_idx] = sorted_lattice[rank]

    return mapping