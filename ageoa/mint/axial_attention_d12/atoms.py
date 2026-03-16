from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from typing import TYPE_CHECKING

import icontract  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import torch
from ageoa.ghost.registry import register_atom
from .witnesses import witness_rowselfattention

@register_atom(witness_rowselfattention)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda self_attn_mask: self_attn_mask is not None, "self_attn_mask cannot be None")  # type: ignore[untyped-decorator]
@icontract.require(lambda self_attn_padding_mask: self_attn_padding_mask is not None, "self_attn_padding_mask cannot be None")  # type: ignore[untyped-decorator]
@icontract.ensure(lambda result: result is not None, "RowSelfAttention output must not be None")  # type: ignore[untyped-decorator]
def row_self_attention(x: torch.Tensor, self_attn_mask: torch.Tensor | None, self_attn_padding_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies row-wise self-attention, a mechanism that computes pairwise similarity scores across row positions to produce context-aware representations. Each row attends to all other rows, weighted by learned relevance.

    Args:
        x: Input tensor to attend over.
        self_attn_mask: Mask controlling which positions can attend to each other.
        self_attn_padding_mask: Mask indicating padded positions to ignore.

    Returns:
        Attention-weighted output tensor with same shape as input.
    """
    import torch
    # Row-wise self-attention in torch
    if x.dim() == 2:
        seq_len, dim = x.shape
        scale = dim ** 0.5
        scores = torch.matmul(x, x.T) / scale
        if self_attn_mask is not None:
            scores = scores.masked_fill(self_attn_mask == 0, float('-inf'))
        if self_attn_padding_mask is not None:
            scores = scores.masked_fill(self_attn_padding_mask.bool().unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return (torch.matmul(attn, x), attn)
    # 3D: (B, S, D)
    B, S, D = x.shape
    scale = D ** 0.5
    scores = torch.bmm(x, x.transpose(1, 2)) / scale
    if self_attn_mask is not None:
        scores = scores.masked_fill(self_attn_mask == 0, float('-inf'))
    if self_attn_padding_mask is not None:
        scores = scores.masked_fill(self_attn_padding_mask.bool().unsqueeze(1), float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return (torch.bmm(attn, x), attn)
