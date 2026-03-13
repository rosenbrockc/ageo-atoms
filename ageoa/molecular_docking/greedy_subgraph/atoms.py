from __future__ import annotations
from typing import Any
Boolean: Any = Any
Node: Any = Any

import networkx as nx
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_greedy_maximum_subgraph
from ageoa.ghost.abstract import Boolean, Node


@register_atom(witness_greedy_maximum_subgraph)
@icontract.require(lambda adjacency: adjacency.ndim >= 1, "adjacency must have at least one dimension")
@icontract.require(lambda scores: scores.ndim >= 1, "scores must have at least one dimension")
@icontract.require(lambda adjacency: adjacency is not None, "adjacency cannot be None")
@icontract.require(lambda adjacency: isinstance(adjacency, np.ndarray), "adjacency must be np.ndarray")
@icontract.require(lambda scores: scores is not None, "scores cannot be None")
@icontract.require(lambda scores: isinstance(scores, np.ndarray), "scores must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def greedy_maximum_subgraph(adjacency: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Greedily selects a maximum-weight subgraph from a molecular interaction graph by iteratively adding the highest-scoring connected node.

    Args:
        adjacency: Adjacency matrix of the molecular interaction graph, shape (n_nodes, n_nodes)
        scores: Node affinity scores used to guide the greedy selection, shape (n_nodes,)

    Returns:
        Boolean mask of selected subgraph nodes, shape (n_nodes,)
    """
    raise NotImplementedError("Wire to original implementation")