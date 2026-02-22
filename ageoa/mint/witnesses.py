"""Ghost witnesses."""

from ageoa.ghost.abstract import AbstractArray

def witness_axial_attention(data: AbstractArray) -> AbstractArray:
    """Witness for axial_attention."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_rotary_positional_embeddings(data: AbstractArray) -> AbstractArray:
    """Witness for rotary_positional_embeddings."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)
