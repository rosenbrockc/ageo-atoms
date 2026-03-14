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
from .witnesses import witness_compute_topo_loss, witness_topological_loss_computation
from typing import TypeAlias

from jax import numpy as jnp

Array: TypeAlias = jnp.ndarray
Scalar: TypeAlias = float

@register_atom(witness_topological_loss_computation)
@icontract.require(lambda tau: isinstance(tau, (float, int, np.number)), "tau must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "topological_loss_computation output must not be None")
def topological_loss_computation(key: Array, logits: Array, pos32: Array, nbr_idx: Array, b: Array, max_iters: int, tau: float) -> Scalar:
    """Computes a loss measuring how well a grouping algorithm links nearby objects into clusters. Uses model scores, positions, and neighbor lists.

    Args:
        key: random number generator key
        logits: model output scores
        pos32: positional data
        nbr_idx: neighbor indices
        b: loss weight parameter
        max_iters: maximum iterations
        tau: temperature or scaling parameter

    Returns:
        the computed loss value
    """
    raise NotImplementedError("Wire to original implementation")