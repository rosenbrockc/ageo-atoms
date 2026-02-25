"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_encodedistancematrix)
@icontract.require(lambda mat_list: isinstance(mat_list, np.ndarray), "mat_list must be a numpy array")
@icontract.ensure(lambda result, **kwargs: result is not None, "EncodeDistanceMatrix output must not be None")
def encodedistancematrix(mat_list: List[np.ndarray], max_cdr3: int, max_epi: int) -> np.ndarray:
    """Takes a list of matrices and pads them to a specified maximum dimension, effectively creating a batched and padded distance matrix representation.

    Args:
        mat_list: A list of 2D numpy arrays (matrices) to be encoded.
        max_cdr3: The maximum size for the first dimension to pad to.
        max_epi: The maximum size for the second dimension to pad to.

    Returns:
        A single numpy array containing the padded and stacked matrices.
    """
    raise NotImplementedError("Wire to original implementation")
