from __future__ import annotations
from typing import Any
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import scipy.spatial
from .witnesses import witness_voronoitessellation, witness_delaunaytriangulation

@register_atom(witness_voronoitessellation)  # type: ignore[untyped-decorator]
@icontract.require(lambda points: points is not None, "points cannot be None")
@icontract.ensure(lambda result: result is not None, "VoronoiTessellation output must not be None")
def voronoitessellation(
    points: np.ndarray,
    furthest_site: bool = False,
    incremental: bool = False,
    qhull_options: str | None = None,
) -> "scipy.spatial.Voronoi":
    """Computes the Voronoi diagram for a set of input points using Qhull, partitioning space into regions closest to each point. Supports incremental construction and custom Qhull options.

    Args:
        points: N >= D+1 for non-degenerate tessellation
        furthest_site: optional, default False
        incremental: optional, default False
        qhull_options: optional, default None

    Returns:
        Voronoi tessellation result
    """
    return scipy.spatial.Voronoi(
        points,
        furthest_site=furthest_site,
        incremental=incremental,
        qhull_options=qhull_options,
    )

@register_atom(witness_delaunaytriangulation)  # type: ignore[untyped-decorator]
@icontract.require(lambda points: points is not None, "points cannot be None")
@icontract.ensure(lambda result: result is not None, "DelaunayTriangulation output must not be None")
def delaunaytriangulation(
    points: np.ndarray,
    furthest_site: bool = False,
    incremental: bool = False,
    qhull_options: str | None = None,
) -> "scipy.spatial.Delaunay":
    """Computes the Delaunay triangulation for a set of input points using Qhull, producing a simplex mesh where no point lies inside the circumsphere of any simplex. Supports incremental construction and custom Qhull options.

    Args:
        points: N >= D+1 for non-degenerate triangulation
        furthest_site: optional, default False
        incremental: optional, default False
        qhull_options: optional, default None

    Returns:
        dual graph of Voronoi diagram for same point set
    """
    return scipy.spatial.Delaunay(
        points,
        furthest_site=furthest_site,
        incremental=incremental,
        qhull_options=qhull_options,
    )
