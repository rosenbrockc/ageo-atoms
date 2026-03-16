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
@icontract.require(lambda mode: mode is not None, "mode cannot be None")
@icontract.require(lambda max_neighbors: max_neighbors is not None, "max_neighbors cannot be None")
@icontract.require(lambda batch_size: batch_size is not None, "batch_size cannot be None")
@icontract.ensure(lambda result: result is not None, "find_fof_clusters output must not be None")
def find_fof_clusters(x: np.ndarray, b: float, L: float, mode: str, max_neighbors: int, batch_size: int) -> np.ndarray:
    """Computes friends-of-friends (FOF) clusters for a set of points on a periodic grid. This is a core clustering algorithm used in cosmology and astrophysics to identify gravitationally bound structures.

    Args:
        x: Must be a numeric array.
        b: Must be a positive float.
        L: Must be a positive float.
        mode: Input data.
        max_neighbors: Must be a positive integer.
        batch_size: Must be a positive integer.

    Returns:
        The output array will have a length equal to the number of particles.
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