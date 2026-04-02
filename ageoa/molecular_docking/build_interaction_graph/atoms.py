from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_networkx_weighted_graph_materialization, witness_pair_distance_compatibility_check, witness_weighted_interaction_edge_derivation

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_pair_distance_compatibility_check)
@icontract.require(lambda interaction_distance: isinstance(interaction_distance, (float, int, np.number)), "interaction_distance must be numeric")
@icontract.ensure(lambda result: result is not None, "Pair Distance Compatibility Check output must not be None")
def pair_distance_compatibility_check(L_feature_min_max: object, R_features_distance: object, interaction_distance: float) -> bool:
    """Evaluates whether a candidate left/right feature pair satisfies interaction-distance constraints.

    Args:
        L_feature_min_max: Expected to encode min/max distance envelope for left feature context.
        R_features_distance: Distance values for right-side feature candidates.
        interaction_distance: Non-negative interaction threshold/target distance.

    Returns:
        True when pair satisfies distance constraints.
    """
    l_min_max = np.asarray(L_feature_min_max, dtype=float)
    r_dist = np.asarray(R_features_distance, dtype=float)
    if l_min_max.size == 0 or r_dist.size == 0:
        return False
    # Treat the left feature distances as an interval and accept when any
    # right-side distance falls within the interaction-distance-expanded band.
    l_min = float(l_min_max.min())
    l_max = float(l_min_max.max())
    lower = l_min - float(interaction_distance)
    upper = l_max + float(interaction_distance)
    return bool(np.any((r_dist >= lower) & (r_dist <= upper)))

@register_atom(witness_weighted_interaction_edge_derivation)
@icontract.require(lambda interaction_distance: isinstance(interaction_distance, (float, int, np.number)), "interaction_distance must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "Weighted Interaction Edge Derivation all outputs must not be None")
def weighted_interaction_edge_derivation(L_features: object, R_features: object, L_distance_matrix: object, R_distance_matrix: object, interaction_distance: float, distance_match: bool) -> tuple[list[tuple[object, object, float]], list[object] | set[object]]:
    """
    Args:
        L_features: Left-side feature set.
        R_features: Right-side feature set.
        L_distance_matrix: Pairwise distances among left-side features.
        R_distance_matrix: Pairwise distances among right-side features.
        interaction_distance: Distance criterion used for matching/scoring.
        distance_match: Compatibility signal from pair distance check.

    Returns:
        edges: Each edge contains endpoints and associated weight.
        nodes: Unique nodes participating in edges.
    """
    L_feat = list(L_features) if not isinstance(L_features, list) else L_features
    R_feat = list(R_features) if not isinstance(R_features, list) else R_features
    L_dm = np.asarray(L_distance_matrix)
    R_dm = np.asarray(R_distance_matrix)
    edges: list[tuple[object, object, float]] = []
    node_set: set[object] = set()

    if not distance_match:
        return edges, list(node_set)

    for i, l_feat in enumerate(L_feat):
        for j, r_feat in enumerate(R_feat):
            # Weight based on how close the distance matrices match
            l_dists = L_dm[i] if L_dm.ndim > 1 else L_dm
            r_dists = R_dm[j] if R_dm.ndim > 1 else R_dm
            # Compute a compatibility score as inverse distance difference
            weight = max(0.0, interaction_distance - abs(float(np.mean(l_dists)) - float(np.mean(r_dists))))
            if weight > 0:
                node_pair = (l_feat, r_feat)
                edges.append((node_pair[0], node_pair[1], weight))
                node_set.add(node_pair[0])
                node_set.add(node_pair[1])

    return edges, list(node_set)

@register_atom(witness_networkx_weighted_graph_materialization)
@icontract.require(lambda edges: edges is not None, "edges cannot be None")
@icontract.require(lambda nodes: nodes is not None, "nodes cannot be None")
@icontract.ensure(lambda result: result is not None, "NetworkX Weighted Graph Materialization output must not be None")
# Invalid pseudo-signature removed; keep the typed definition below.
def networkx_weighted_graph_materialization(edges: list[tuple[object, object, float]], nodes: list[object] | set[object]) -> nx.Graph | nx.DiGraph:
    """
    Args:
        nodes: Node IDs should be hashable.

    Returns:
        Contains all provided nodes and weighted edges.
    """
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G
