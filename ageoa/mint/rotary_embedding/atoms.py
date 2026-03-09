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
    raise NotImplementedError("Wire to original implementation")
