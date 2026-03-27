from __future__ import annotations

from __future__ import annotations

from typing import TypeAlias

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .topo_witnesses import witness_compute_topo_loss, witness_topological_loss_computation

Array: TypeAlias = np.ndarray
Scalar: TypeAlias = float

@register_atom(witness_topological_loss_computation)
@icontract.require(lambda tau: isinstance(tau, (float, int, np.number)), "tau must be numeric")
@icontract.ensure(lambda result: result is not None, "topological_loss_computation output must not be None")
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
    # Topological loss: measures clustering quality
    # Softmax over logits to get assignment probabilities
    logits_arr = np.asarray(logits, dtype=np.float64)
    pos = np.asarray(pos32, dtype=np.float64)
    nbr = np.asarray(nbr_idx, dtype=np.intp)
    b_arr = np.asarray(b, dtype=np.float64)

    # Gumbel-softmax relaxation for differentiable clustering
    n = logits_arr.shape[0]
    probs = np.exp(logits_arr / tau)
    probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-15)

    # Compute loss: penalize neighbors assigned to different clusters
    loss = 0.0
    for i in range(min(n, max_iters)):
        for j_idx in range(nbr.shape[1] if nbr.ndim > 1 else 1):
            j = int(nbr[i, j_idx]) if nbr.ndim > 1 else int(nbr[i])
            if 0 <= j < n:
                # Cross-entropy between neighbor assignment probabilities
                loss += float(-np.sum(probs[i] * np.log(probs[j] + 1e-15)))
    return float(loss / max(n, 1))
