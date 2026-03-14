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
    raise NotImplementedError("Wire to original implementation")