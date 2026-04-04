"""Deterministic wrappers for selected mint attention operations."""

from __future__ import annotations

import icontract
import numpy as np

from ageoa.ghost.registry import register_atom

from .witnesses import witness_axial_attention, witness_rotary_positional_embeddings


def _softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((-x2, x1), axis=-1)


def _apply_rotary_pos_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return (x * cos) + (_rotate_half(x) * sin)


def _is_finite_ndarray(value: object) -> bool:
    return isinstance(value, np.ndarray) and np.isfinite(value).all()


def _is_valid_axial_input(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 4
        and value.shape[0] > 0
        and value.shape[1] > 0
        and value.shape[-1] > 0
        and np.isfinite(value).all()
    )


def _is_valid_rotary_input(value: object) -> bool:
    return isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-1] % 2 == 0 and np.isfinite(value).all()


def _same_shape(a: object, b: object) -> bool:
    return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape


@register_atom(witness_axial_attention)
@icontract.require(lambda x: x is not None, "x must not be None")
@icontract.require(lambda x: _is_valid_axial_input(x), "x must be a finite numpy array with shape (rows, cols, batch, embed_dim)")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, tuple), "result must be a tuple")
def axial_attention(
    x: np.ndarray,
    self_attn_mask: np.ndarray | None = None,
    self_attn_padding_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a deterministic row-wise self-attention approximation.

    Args:
        x: Input tensor with shape `(rows, cols, batch, embed_dim)`.
        self_attn_mask: Optional attention mask. Present for API compatibility.
        self_attn_padding_mask: Optional padding mask with shape `(batch, rows, cols)`.

    Returns:
        output: Contextualized tensor with the same shape as `x`.
        attn_probs: Attention weights over the column dimension.
    """
    del self_attn_mask

    rows, cols, batch_size, embed_dim = x.shape
    scaling = 1.0 / np.sqrt(float(max(embed_dim * rows, 1)))
    q = np.transpose(x, (0, 2, 1, 3))
    k = np.transpose(x, (0, 2, 1, 3))
    v = np.transpose(x, (0, 2, 1, 3))

    attn_logits = np.einsum("rbid,rbjd->bij", q, k) * scaling
    if self_attn_padding_mask is not None:
        padding = np.asarray(self_attn_padding_mask, dtype=bool)
        if padding.shape == (batch_size, rows, cols):
            masked_cols = padding[:, 0, :]
            attn_logits = np.where(masked_cols[None, :, None], -1e9, attn_logits)
    attn_probs = _softmax(attn_logits, axis=-1)
    context = np.einsum("bij,rbjd->rbid", attn_probs, v)
    output = np.transpose(context, (0, 2, 1, 3))
    return output.astype(x.dtype, copy=False), attn_probs.astype(x.dtype, copy=False)


@register_atom(witness_rotary_positional_embeddings)
@icontract.require(lambda q: q is not None, "q must not be None")
@icontract.require(lambda k: k is not None, "k must not be None")
@icontract.require(lambda q: _is_valid_rotary_input(q), "q must be a finite numpy array with at least two dimensions and an even embedding size")
@icontract.require(lambda k: _is_valid_rotary_input(k), "k must be a finite numpy array with at least two dimensions and an even embedding size")
@icontract.require(lambda q, k: _same_shape(q, k), "q and k must have identical shapes")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, tuple), "result must be a tuple")
def rotary_positional_embeddings(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply rotary position embeddings to a query/key pair.

    Args:
        q: Query tensor with an even embedding dimension.
        k: Key tensor with the same shape as `q`.

    Returns:
        q_rotated: Query tensor after rotary position embedding.
        k_rotated: Key tensor after rotary position embedding.
    """
    seq_len = k.shape[-2]
    dim = k.shape[-1]
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    positions = np.arange(seq_len, dtype=np.float64)
    freqs = np.einsum("i,j->ij", positions, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(emb)[None, ...]
    sin = np.sin(emb)[None, ...]
    return (
        _apply_rotary_pos_emb(q, cos, sin).astype(q.dtype, copy=False),
        _apply_rotary_pos_emb(k, cos, sin).astype(k.dtype, copy=False),
    )
