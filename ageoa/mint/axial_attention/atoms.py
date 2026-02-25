"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Any
import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_rowselfattention)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda self_attn_mask: self_attn_mask is not None, "self_attn_mask cannot be None")
@icontract.require(lambda self_attn_padding_mask: self_attn_padding_mask is not None, "self_attn_padding_mask cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "RowSelfAttention output must not be None")
def rowselfattention(x: Any, self_attn_mask: Any, self_attn_padding_mask: Any) -> Any:
    """Opaque DL boundary: RowSelfAttention

    Args:
        x: Input data.
        self_attn_mask: Input data.
        self_attn_padding_mask: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")