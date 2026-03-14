from __future__ import annotations
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

import scipy.spatial
from .witnesses import witness_voronoitessellation, witness_delaunaytriangulation

@register_atom(witness_voronoitessellation)  # type: ignore[untyped-decorator]
@icontract.require(lambda incremental: incremental is not None, "incremental cannot be None")
@icontract.require(lambda qhull_options: qhull_options is not None, "qhull_options cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "VoronoiTessellation output must not be None")
def voronoitessellation(points: np.ndarray, incremental: bool, qhull_options: str) -> "scipy.spatial.Voronoi":
    """Computes the Voronoi diagram for a set of input points using Qhull, partitioning space into regions closest to each point. Supports incremental construction and custom Qhull options.

    Args:
        points: N >= D+1 for non-degenerate tessellation
        incremental: optional, default False
        qhull_options: optional, default empty string

    Returns:
        Voronoi tessellation result
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_delaunaytriangulation)  # type: ignore[untyped-decorator]
@icontract.require(lambda incremental: incremental is not None, "incremental cannot be None")
@icontract.require(lambda qhull_options: qhull_options is not None, "qhull_options cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "DelaunayTriangulation output must not be None")
def delaunaytriangulation(points: np.ndarray, incremental: bool, qhull_options: str) -> "scipy.spatial.Delaunay":
    """Computes the Delaunay triangulation for a set of input points using Qhull, producing a simplex mesh where no point lies inside the circumsphere of any simplex. Supports incremental construction and custom Qhull options.

    Args:
        points: N >= D+1 for non-degenerate triangulation
        incremental: optional, default False
        qhull_options: optional, default empty string

    Returns:
        dual graph of Voronoi diagram for same point set
    """
    raise NotImplementedError("Wire to original implementation")