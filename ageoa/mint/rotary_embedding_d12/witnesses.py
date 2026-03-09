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

def witness_rotaryembedding(q: AbstractArray, k: AbstractArray) -> AbstractArray:
    """Ghost witness for opaque boundary: RotaryEmbedding."""
    # RotaryEmbedding applies in-place sinusoidal rotation to q and k.
    # Shape is fully preserved: no projection, pooling, or sequence change.
    # Typical layout: B=batch, H=num_heads, N=seq_len, D=head_dim.
    #
    # Under vmap the outermost batch dim B is stripped by the transform;
    # q and k then arrive as (H, N, D). The witness stays correct because
    # it mirrors q.shape and k.shape symbolically regardless of rank.
    #
    # Precondition: q.shape == k.shape (standard RoPE contract).
    q_out = AbstractArray(q.shape, dtype='float32')
    k_out = AbstractArray(k.shape, dtype='float32')
    return (q_out, k_out)
