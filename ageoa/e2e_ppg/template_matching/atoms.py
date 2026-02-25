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
witness_templatefeaturecomputation = object()
@register_atom(witness_templatefeaturecomputation)  # type: ignore[untyped-decorator]
@icontract.require(lambda hc: hc is not None, "hc cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "TemplateFeatureComputation output must not be None")
def templatefeaturecomputation(hc: object) -> object:
    """Computes template-matching features from the provided input without persistent state mutation.

    Args:
        hc: Required input context for feature computation.

    Returns:
        Derived deterministically from hc.
    """
    raise NotImplementedError("Wire to original implementation")