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
from .witnesses import witness_compute_topo_loss
from typing import TypeAlias

from jax import numpy as jnp

Array: TypeAlias = jnp.ndarray
Scalar: TypeAlias = float
witness_topological_loss_computation = lambda: True

@register_atom(witness_topological_loss_computation)
@icontract.require(lambda tau: isinstance(tau, (float, int, np.number)), "tau must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "topological_loss_computation output must not be None")
def topological_loss_computation(key: Array, logits: Array, pos32: Array, nbr_idx: Array, b: Array, max_iters: int, tau: float) -> Scalar:
    """Computes a topological loss using logits, positional data, and neighborhood indices. The process is stochastic and controlled by a random key.

    Args:
        key: A JAX random number generator key for stochastic operations.
        logits: Logits output from a model.
        pos32: 32-bit positional data.
        nbr_idx: Neighbor indices.
        b: A parameter for the loss computation.
        max_iters: Maximum number of iterations for the computation.
        tau: A temperature or scaling parameter.

    Returns:
        The computed topological loss value.
    """
    raise NotImplementedError("Wire to original implementation")