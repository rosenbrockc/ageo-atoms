from __future__ import annotations

import numpy as np

from ageoa.sklearn.images import (
    extract_patches_2d,
    grid_to_graph,
    img_to_graph,
    reconstruct_from_patches_2d,
)


def test_sklearn_image_atoms_smoke() -> None:
    image2d = np.arange(16).reshape(4, 4)
    patches = extract_patches_2d(image2d, (2, 2))
    reconstructed = reconstruct_from_patches_2d(patches, (4, 4))

    assert getattr(patches, "shape", None) == (9, 2, 2)
    assert getattr(reconstructed, "shape", None) == (4, 4)

    image3d = np.arange(8, dtype=float).reshape(2, 2, 2)
    graph_from_image = img_to_graph(image3d)
    graph_from_grid = grid_to_graph(2, 2, 2)

    assert getattr(graph_from_image, "shape", None) == (8, 8)
    assert getattr(graph_from_grid, "shape", None) == (8, 8)
