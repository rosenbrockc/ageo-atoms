from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_axial_attention(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for axial attention. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)


def witness_rotary_positional_embeddings(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for rotary positional embeddings. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)
