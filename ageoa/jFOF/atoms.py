from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""
from typing import Any

import typing
import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_find_fof_clusters
# Witness functions should be imported from the generated witnesses module

@register_atom(witness_find_fof_clusters)
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda b: b is not None, "b cannot be None")
@icontract.require(lambda L: L is not None, "L cannot be None")
@icontract.ensure(lambda result: result is not None, "find_fof_clusters output must not be None")
def find_fof_clusters(
    x: np.ndarray,
    b: float,
    L: float,
    mode: str = "precompute",
    max_neighbors: int | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """Compute periodic friends-of-friends cluster labels for a point cloud.

    Args:
        x: Particle positions with shape ``(n_particles, n_dims)``.
        b: Linking length threshold.
        L: Periodic box size used when ``mode`` is ``"periodic"``.
        mode: Neighbor-search mode carried through from the vendored wrapper.
        max_neighbors: Optional neighborhood-allocation hint from the vendored wrapper.
        batch_size: Optional batching hint from the vendored wrapper.

    Returns:
        Cluster label for each particle.
    """
    # Friends-of-Friends clustering on periodic grid
    from scipy.spatial import cKDTree
    n = x.shape[0]
    if n == 0:
        return np.array([], dtype=np.intp)
    # Handle periodic boundary conditions by wrapping
    if mode == 'periodic' and L > 0:
        tree = cKDTree(x, boxsize=L)
    else:
        tree = cKDTree(x)
    # Find all pairs within linking length b
    pairs = tree.query_pairs(r=b, output_type='ndarray')
    # Union-Find to build clusters
    labels = np.arange(n, dtype=np.intp)
    def find(i):
        while labels[i] != i:
            labels[i] = labels[labels[i]]
            i = labels[i]
        return i
    def union(a, c):
        ra, rc = find(a), find(c)
        if ra != rc:
            labels[ra] = rc
    for i, j in pairs:
        union(i, j)
    # Compress
    for i in range(n):
        labels[i] = find(i)
    return labels
