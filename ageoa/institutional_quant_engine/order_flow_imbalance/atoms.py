"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_orderflowimbalanceevaluation)  # type: ignore[untyped-decorator, name-defined]
@icontract.require(lambda row: row is not None, "row cannot be None")
@icontract.require(lambda prev_row: prev_row is not None, "prev_row cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "OrderFlowImbalanceEvaluation output must not be None")
def orderflowimbalanceevaluation(row: object, prev_row: object) -> float:
    """Computes the order flow imbalance signal for the current observation relative to the previous observation as a pure, stateless transformation.

    Args:
        row: Must contain the fields required for OFI computation.
        prev_row: Represents the immediately preceding row; schema-compatible with row.

    Returns:
        Deterministic scalar derived only from row and prev_row.
    """
    raise NotImplementedError("Wire to original implementation")