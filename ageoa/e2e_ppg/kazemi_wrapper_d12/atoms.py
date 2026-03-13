from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_normalizesignal, witness_wrapperevaluate
from typing import Any

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_normalizesignal)
@icontract.require(lambda arr: arr is not None, "arr cannot be None")
def normalizesignal(arr: np.ndarray) -> np.ndarray:
    """Normalizes a raw array to a standard scale, producing a unit-normalized output suitable for downstream comparison or scoring.

    Args:
        arr: non-empty, finite values

    Returns:
        same shape as input, normalized
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_wrapperevaluate)
@icontract.require(lambda prediction: prediction is not None, "prediction cannot be None")
@icontract.require(lambda raw_signal: raw_signal is not None, "raw_signal cannot be None")
@icontract.require(lambda normalized_arr: normalized_arr is not None, "normalized_arr cannot be None")
def wrapperevaluate(prediction: Any, raw_signal: np.ndarray, normalized_arr: np.ndarray) -> Any:
    """
    Args:
        prediction: Input data.
        raw_signal: non-empty, finite values
        normalized_arr: output of NormalizeSignal

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")
