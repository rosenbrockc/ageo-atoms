from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import witness_rotaryembedding


@register_atom(witness_rotaryembedding)
@icontract.require(lambda q: q.ndim >= 2, "q must be at least 2-D")
@icontract.require(lambda k: k.ndim >= 2, "k must be at least 2-D")
@icontract.require(lambda q: isinstance(q, np.ndarray), "q must be np.ndarray")
@icontract.require(lambda k: isinstance(k, np.ndarray), "k must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a 2-tuple")
def rotaryembedding(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Positional Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor with shape (..., seq_len, head_dim).
        k: Key tensor with shape (..., seq_len, head_dim).

    Returns:
        Tuple of rotated (query, key) tensors with same shapes as inputs.
    """
    # RoPE: rotate query and key by position-dependent angles
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    # Compute frequency bands
    freqs = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, freqs)  # (seq_len, head_dim/2)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # Rotate pairs of dimensions
    def _rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = np.empty_like(x)
        rotated[..., 0::2] = x1 * cos_a - x2 * sin_a
        rotated[..., 1::2] = x1 * sin_a + x2 * cos_a
        return rotated
    return (_rotate(q), _rotate(k))
