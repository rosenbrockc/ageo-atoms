from __future__ import annotations
from typing import Any
import icontract
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
from ageoa.ghost.registry import register_atom
from .witnesses import witness_pinlikelihoodevaluator
from ageoa.ghost.registry import register_atom
from .witnesses import witness_pinlikelihoodevaluator

# Witness functions should be imported from the generated witnesses module
@register_atom(witness_pinlikelihoodevaluator)  # type: ignore[name-defined, untyped-decorator]
@register_atom(witness_pinlikelihoodevaluator)
@icontract.require(lambda params: params is not None, "params cannot be None")
@icontract.require(lambda B: B is not None, "B cannot be None")
@icontract.require(lambda S: S is not None, "S cannot be None")
def pinlikelihoodevaluator(params: dict[str, Any], B: float | np.ndarray[Any, np.dtype[Any]], S: float | np.ndarray[Any, np.dtype[Any]]) -> float:
    """Evaluates the log-likelihood (or likelihood) of observed data given model parameters and sufficient statistics B and S. This is a stateless, pure oracle computation with no side effects or persistent state.

Args:
    params: must be valid parameter configuration for the Probability of Informed Trading (PIN) model
    B: non-negative
    S: non-negative

Returns:
    real-valued; -inf indicates zero likelihood"""
    raise NotImplementedError("Wire to original implementation")