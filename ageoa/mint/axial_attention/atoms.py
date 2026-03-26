from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
from typing import Any
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_rowselfattention

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_rowselfattention)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda self_attn_mask: self_attn_mask is not None, "self_attn_mask cannot be None")
@icontract.require(lambda self_attn_padding_mask: self_attn_padding_mask is not None, "self_attn_padding_mask cannot be None")
@icontract.ensure(lambda result: result is not None, "RowSelfAttention output must not be None")
def rowselfattention(x: np.ndarray, self_attn_mask: np.ndarray, self_attn_padding_mask: np.ndarray) -> np.ndarray:
    """Applies row-wise self-attention, a mechanism that computes pairwise similarity scores across row positions to produce context-aware representations. Each row attends to all other rows, weighted by learned relevance.

    Args:
        x: Input tensor to attend over.
        self_attn_mask: Mask controlling which positions can attend to each other.
        self_attn_padding_mask: Mask indicating padded positions to ignore.

    Returns:
        Attention-weighted output tensor with same shape as input.
    """
    # Row-wise self-attention in numpy
    # x shape: (batch, seq_len, dim) or (seq_len, dim)
    if x.ndim == 2:
        seq_len, dim = x.shape
        scale = np.sqrt(dim)
        scores = x @ x.T / scale
        # Apply masks
        if self_attn_mask is not None:
            scores = np.where(self_attn_mask != 0, scores, -1e9)
        if self_attn_padding_mask is not None:
            scores = np.where(~self_attn_padding_mask.astype(bool).reshape(-1, 1), scores, -1e9)
        scores -= scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores) / (np.exp(scores).sum(axis=-1, keepdims=True) + 1e-15)
        return attn @ x
    # 3D: batch attention
    B, S, D = x.shape
    scale = np.sqrt(D)
    scores = np.matmul(x, x.transpose(0, 2, 1)) / scale
    if self_attn_mask is not None:
        scores = np.where(self_attn_mask != 0, scores, -1e9)
    if self_attn_padding_mask is not None:
        mask = self_attn_padding_mask.astype(bool)
        if mask.ndim == 2:
            scores = np.where(~mask[:, np.newaxis, :], scores, -1e9)
    scores -= scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores) / (np.exp(scores).sum(axis=-1, keepdims=True) + 1e-15)
    return np.matmul(attn, x)
