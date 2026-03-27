from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_topological_loss_computation(key: AbstractArray, logits: AbstractArray, pos32: AbstractArray, nbr_idx: AbstractArray, b: AbstractArray, max_iters: AbstractArray, tau: AbstractArray) -> AbstractArray:
    """Shape-and-type check for topological loss computation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=key.shape,
        dtype="float64",
    )
    return result


def witness_compute_topo_loss(key: AbstractArray, logits: AbstractArray, pos32: AbstractArray, nbr_idx: AbstractArray, b: AbstractArray, max_iters: AbstractScalar, tau: AbstractScalar) -> AbstractScalar:
    """Shape-and-type check for compute topo loss. Returns output metadata without running the real computation."""
    result = AbstractScalar(
        dtype="float64",
    )
    return result
