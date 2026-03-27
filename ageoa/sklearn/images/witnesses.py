"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractArray
except ImportError:
    pass


def witness_image_patch_sampling_and_assembly(
    image: AbstractArray,
    patch_size: AbstractArray,
    max_patches: AbstractArray,
    random_state: AbstractArray,
) -> AbstractArray:
    """Ghost witness for image_patch_sampling_and_assembly (query, state-preserving)."""
    return AbstractArray(shape=image.shape, dtype=image.dtype)


def witness_patches_to_image_reconstruction(
    patches: AbstractArray,
    image_size: AbstractArray,
) -> AbstractArray:
    """Ghost witness for patches_to_image_reconstruction (query, state-preserving)."""
    return AbstractArray(shape=patches.shape, dtype=patches.dtype)


def witness_3d_image_graph_materialization(
    img: AbstractArray,
    mask: AbstractArray,
    return_as: AbstractArray,
    dtype: AbstractArray,
) -> AbstractArray:
    """Ghost witness for 3D Image Graph Materialization (query, state-preserving)."""
    return AbstractArray(shape=img.shape, dtype=img.dtype)


def witness_voxel_grid_graph_assembly(
    n_x: AbstractArray,
    n_y: AbstractArray,
    n_z: AbstractArray,
    mask: AbstractArray,
    return_as: AbstractArray,
    dtype: AbstractArray,
) -> AbstractArray:
    """Ghost witness for Voxel Grid Graph Assembly (query, state-preserving)."""
    return AbstractArray(shape=n_x.shape, dtype=n_x.dtype)
