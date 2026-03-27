"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

# mypy: disable-error-code=untyped-decorator

from typing import Any

import importlib

import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import (
    witness_3d_image_graph_materialization,
    witness_image_patch_sampling_and_assembly,
    witness_patches_to_image_reconstruction,
    witness_voxel_grid_graph_assembly,
)

_SCIONA_UNSET = object()

_SCIONA_SOURCE_MODULE = importlib.import_module("sklearn.feature_extraction.image")
_EXTRACT_PATCHES_2D: Any = getattr(_SCIONA_SOURCE_MODULE, "extract_patches_2d")
_RECONSTRUCT_FROM_PATCHES_2D: Any = getattr(
    _SCIONA_SOURCE_MODULE,
    "reconstruct_from_patches_2d",
)
_IMG_TO_GRAPH: Any = getattr(_SCIONA_SOURCE_MODULE, "img_to_graph")
_GRID_TO_GRAPH: Any = getattr(_SCIONA_SOURCE_MODULE, "grid_to_graph")


@register_atom(witness_image_patch_sampling_and_assembly)
@icontract.require(lambda image: image is not None, "image cannot be None")
@icontract.require(lambda patch_size: patch_size is not None, "patch_size cannot be None")
@icontract.ensure(
    lambda result: result is not None,
    "image_patch_sampling_and_assembly output must not be None",
)
def extract_patches_2d(
    image: Any,
    patch_size: Any,
    *,
    max_patches: Any = _SCIONA_UNSET,
    random_state: Any = _SCIONA_UNSET,
) -> object:
    """Computes valid patch counts, extracts sliding 2D patches from an image tensor, and optionally subsamples patches when max_patches is specified.

    Args:
        image: Spatial dimensions must be >= patch_size.
        patch_size: Positive patch height and width.
        max_patches: If set, limits number of returned patches; may represent count or fraction depending on implementation.
        random_state: Used only when randomized patch subsampling is active.

    Returns:
        object
    """
    call_kwargs: dict[str, Any] = {}
    if max_patches is not _SCIONA_UNSET:
        call_kwargs["max_patches"] = max_patches
    if random_state is not _SCIONA_UNSET:
        call_kwargs["random_state"] = random_state
    return _EXTRACT_PATCHES_2D(image, patch_size, **call_kwargs)


@register_atom(witness_patches_to_image_reconstruction)
@icontract.require(lambda patches: patches is not None, "patches cannot be None")
@icontract.require(lambda image_size: image_size is not None, "image_size cannot be None")
@icontract.ensure(
    lambda result: result is not None,
    "patches_to_image_reconstruction output must not be None",
)
def reconstruct_from_patches_2d(patches: Any, image_size: Any) -> object:
    """Reconstructs a full 2D image tensor by assembling provided patches into the requested image size.

    Args:
        patches: Contains 2D image patches arranged in a layout compatible with reconstruction.
        image_size: Defines target reconstructed 2D image dimensions.

    Returns:
        object
    """
    return _RECONSTRUCT_FROM_PATCHES_2D(patches, image_size)


@register_atom(witness_3d_image_graph_materialization)
@icontract.require(lambda img: img is not None, "img cannot be None")
@icontract.ensure(
    lambda result: result is not None,
    "3D Image Graph Materialization output must not be None",
)
def img_to_graph(
    img: Any,
    *,
    mask: Any = _SCIONA_UNSET,
    return_as: Any = _SCIONA_UNSET,
    dtype: Any = _SCIONA_UNSET,
) -> object:
    """Converts a 3D image volume into a weighted voxel-adjacency graph by constructing neighborhood edges, computing gradient-based edge weights, optionally masking invalid edges, and returning the graph in the requested representation.

    Args:
        img: shape (n_x, n_y, n_z); numeric dtype
        mask: if provided, broadcast/shape-compatible with img grid
        return_as: supported graph/sparse representation identifier
        dtype: used for edge-weight/output casting

    Returns:
        object
    """
    call_kwargs: dict[str, Any] = {}
    if mask is not _SCIONA_UNSET:
        call_kwargs["mask"] = mask
    if return_as is not _SCIONA_UNSET:
        call_kwargs["return_as"] = return_as
    if dtype is not _SCIONA_UNSET:
        call_kwargs["dtype"] = dtype
    return _IMG_TO_GRAPH(img, **call_kwargs)


@register_atom(witness_voxel_grid_graph_assembly)
@icontract.require(lambda n_x: n_x is not None, "n_x cannot be None")
@icontract.require(lambda n_y: n_y is not None, "n_y cannot be None")
@icontract.ensure(
    lambda result: result is not None,
    "Voxel Grid Graph Assembly output must not be None",
)
def grid_to_graph(
    n_x: Any,
    n_y: Any,
    n_z: Any = _SCIONA_UNSET,
    *,
    mask: Any = _SCIONA_UNSET,
    return_as: Any = _SCIONA_UNSET,
    dtype: Any = _SCIONA_UNSET,
) -> object:
    """Constructs a 3D grid adjacency graph by generating lattice edges, optionally computing gradient-based edge weights, optionally masking invalid connections, and returning the graph in the requested representation.

    Args:
        n_x: positive grid size along x-axis
        n_y: positive grid size along y-axis
        n_z: positive grid size along z-axis
        mask: optional voxel inclusion mask matching grid shape
        return_as: target graph container/class
        dtype: numeric dtype for graph weights

    Returns:
        object
    """
    call_kwargs: dict[str, Any] = {}
    if n_z is not _SCIONA_UNSET:
        call_kwargs["n_z"] = n_z
    if mask is not _SCIONA_UNSET:
        call_kwargs["mask"] = mask
    if return_as is not _SCIONA_UNSET:
        call_kwargs["return_as"] = return_as
    if dtype is not _SCIONA_UNSET:
        call_kwargs["dtype"] = dtype
    return _GRID_TO_GRAPH(n_x, n_y, **call_kwargs)
