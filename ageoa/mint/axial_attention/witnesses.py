"""Auto-generated ghost witnesses for opaque DL boundaries."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractArray
except ImportError:
    pass

def witness_rowselfattention(x: AbstractArray, self_attn_mask: AbstractArray, self_attn_padding_mask: AbstractArray) -> AbstractArray:
    """Ghost witness for opaque boundary: RowSelfAttention."""
    shape = tuple(x.shape)
    *vmapped_dims, n_rows, n_cols, d_model = shape
    out_shape = (*vmapped_dims, n_rows, n_cols, d_model)
    return AbstractArray(shape=out_shape, dtype="float32")
