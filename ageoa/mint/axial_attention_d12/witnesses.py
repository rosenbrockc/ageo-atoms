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
    # x                    : (B, R, C, d_model)  — batch, rows, cols, hidden
    # self_attn_mask        : (C, C) or (B*R, C, C) — additive/bool attention bias
    # self_attn_padding_mask: (B*R, C) or (B, R, C) — True where padded
    #
    # RowSelfAttention attends across the *column* (C) axis for every row
    # independently, so the output tensor is shape-identical to x.
    #
    # vmap note: if this witness is called under jax.vmap over the leading
    # batch axis, x arrives as (R, C, d_model); x.shape still propagates
    # correctly because we read the output shape directly from x.shape.
    
    B_R_C_d = x.shape          # e.g. (B, R, C, d_model) or (R, C, d) under vmap
    return AbstractArray(B_R_C_d, dtype='float32')
