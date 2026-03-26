"""Auto-generated verified atom wrapper."""

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_axial_attention, witness_rotary_positional_embeddings



@register_atom(witness_axial_attention)
@icontract.require(lambda data: np.isfinite(data).all(), "data must contain only finite values")
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty along first axis")
@icontract.require(lambda data: data.ndim >= 2, "data must have at least two dimensions for 2D attention")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 2, "result must have at least two dimensions")
def axial_attention(data: np.ndarray) -> np.ndarray:
    """Implements factorized attention over 2D sequence alignments.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    # Factorized axial attention: row-wise then column-wise softmax attention
    # data shape: (R, C, D) or (R, C)
    if data.ndim == 2:
        R, C = data.shape
        # Row attention: softmax(data @ data.T / sqrt(C)) @ data
        scores_row = data @ data.T / np.sqrt(C)
        scores_row -= scores_row.max(axis=-1, keepdims=True)
        attn_row = np.exp(scores_row) / np.exp(scores_row).sum(axis=-1, keepdims=True)
        out_row = attn_row @ data
        # Column attention
        scores_col = out_row.T @ out_row / np.sqrt(R)
        scores_col -= scores_col.max(axis=-1, keepdims=True)
        attn_col = np.exp(scores_col) / np.exp(scores_col).sum(axis=-1, keepdims=True)
        return (out_row @ attn_col)
    else:
        # 3D+: apply attention along first two axes
        shape = data.shape
        R, C = shape[0], shape[1]
        D = int(np.prod(shape[2:])) if len(shape) > 2 else 1
        flat = data.reshape(R, C, D)
        # Row attention per column
        for c in range(C):
            col_data = flat[:, c, :]  # (R, D)
            scores = col_data @ col_data.T / np.sqrt(D)
            scores -= scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
            flat[:, c, :] = attn @ col_data
        return flat.reshape(shape)

@register_atom(witness_rotary_positional_embeddings)
@icontract.require(lambda data: np.isfinite(data).all(), "data must contain only finite values")
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
@icontract.require(lambda data: data.ndim >= 1, "data must have at least one dimension")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def rotary_positional_embeddings(data: np.ndarray) -> np.ndarray:
    """Encodes relative position into a tensor using rotary transformations.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    # Rotary positional embeddings (RoPE)
    if data.ndim == 1:
        seq_len = data.shape[0]
        d = seq_len
        freqs = 1.0 / (10000 ** (np.arange(0, d, 2, dtype=np.float64) / d))
        positions = np.arange(seq_len, dtype=np.float64)
        angles = np.outer(positions, freqs)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        out = np.empty_like(data)
        half = d // 2
        out[:half] = data[:half] * cos_a[:half, 0] - data[half:2*half] * sin_a[:half, 0] if 2*half <= d else data[:half]
        if 2 * half <= d:
            out[half:2*half] = data[:half] * sin_a[:half, 0] + data[half:2*half] * cos_a[:half, 0]
        return out
    # For higher dims, apply RoPE along last dimension
    shape = data.shape
    seq_len = shape[-2] if data.ndim >= 2 else shape[0]
    head_dim = shape[-1]
    freqs = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, freqs)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x1 = data[..., 0::2]
    x2 = data[..., 1::2]
    out = np.empty_like(data)
    out[..., 0::2] = x1 * cos_a - x2 * sin_a
    out[..., 1::2] = x1 * sin_a + x2 * cos_a
    return out
