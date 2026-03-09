from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from typing import Any

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import *  # type: ignore[import-untyped]

# Witness functions should be imported from the generated witnesses module

@register_atom("witness_hrppipelinerun")  # type: ignore[untyped-decorator]
@icontract.require(lambda data: data is not None, "data cannot be None")
def hrppipelinerun(data: Any) -> Any:
    """Executes the full Hierarchical Risk Parity pipeline: ingests asset return data, constructs a hierarchical clustering structure via a correlation/distance matrix, applies recursive bisection to allocate risk, and emits final portfolio weights.

    Args:
        data: No NaN values; N >= 2; T > N recommended for stable covariance estimation

    Returns:
        All weights in [0, 1]; sum == 1.0
    """
    raise NotImplementedError("Wire to original implementation")
