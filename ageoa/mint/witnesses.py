from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_axial_attention(
    x: AbstractArray,
    self_attn_mask: AbstractArray | None = None,
    self_attn_padding_mask: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for axial attention."""
    del self_attn_mask, self_attn_padding_mask
    cols = x.shape[1]
    return AbstractArray(shape=x.shape, dtype=x.dtype), AbstractArray(shape=(cols, cols), dtype=x.dtype)


def witness_rotary_positional_embeddings(q: AbstractArray, k: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for rotary positional embeddings."""
    return AbstractArray(shape=q.shape, dtype=q.dtype), AbstractArray(shape=k.shape, dtype=k.dtype)
