from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_rowselfattention(x: AbstractArray, self_attn_mask: AbstractArray, self_attn_padding_mask: AbstractArray) -> AbstractArray:
    """Shape-and-type check for opaque boundary: row self attention. Returns output metadata without running the real computation."""
    shape = tuple(x.shape)
    *vmapped_dims, n_rows, n_cols, d_model = shape
    out_shape = (*vmapped_dims, n_rows, n_cols, d_model)
    return AbstractArray(shape=out_shape, dtype="float32")
