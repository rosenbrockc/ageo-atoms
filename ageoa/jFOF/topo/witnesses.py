from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_topological_loss_computation(key: AbstractArray, logits: AbstractArray, pos32: AbstractArray, nbr_idx: AbstractArray, b: AbstractArray, max_iters: AbstractArray, tau: AbstractArray) -> AbstractArray:
    """Ghost witness for topological_loss_computation."""
    result = AbstractArray(
        shape=key.shape,
        dtype="float64",
    )
    return result


def witness_compute_topo_loss(key: AbstractArray, logits: AbstractArray, pos32: AbstractArray, nbr_idx: AbstractArray, b: AbstractArray, max_iters: AbstractScalar, tau: AbstractScalar) -> AbstractScalar:
    """Ghost witness for compute_topo_loss."""
    result = AbstractScalar(
        dtype="float64",
    )
    return result
