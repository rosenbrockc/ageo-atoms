"""Pasqal-inspired molecular docking atoms.

These wrappers are deterministic and self-contained so they can run in tests
without external quantum backends.
"""

from __future__ import annotations

from typing import Dict, List, Set

import icontract
import networkx as nx  # type: ignore

from ageoa.ghost.registry import register_atom
from ageoa.pasqal.docking_state import MolecularDockingState
from ageoa.pasqal.docking_witnesses import (
    witness_graph_transformer,
    witness_quantum_mwis_solver,
    witness_sub_graph_embedder,
)


@register_atom(witness_sub_graph_embedder)
@icontract.require(lambda subgraph_quantity: subgraph_quantity > 0, "subgraph_quantity must be positive")
def sub_graph_embedder(
    current_graph: nx.Graph,
    subgraph_quantity: int,
    state: MolecularDockingState,
) -> tuple[list[dict[int, int]], MolecularDockingState]:
    """Extract deterministic embeddable node mappings from a graph."""
    nodes = sorted(current_graph.nodes())
    limit = min(subgraph_quantity, len(nodes))

    mappings: list[dict[int, int]] = []
    for idx in range(limit):
        mappings.append({int(nodes[idx]): idx})

    new_state = state.model_copy(update={"graph": current_graph})
    return mappings, new_state


@register_atom(witness_graph_transformer)
@icontract.require(lambda mapping: len(mapping) > 0, "mapping must not be empty")
def graph_transformer(
    current_graph: nx.Graph,
    lattice: nx.Graph,
    mapping: dict[int, int],
    state: MolecularDockingState,
) -> tuple[nx.Graph, MolecularDockingState]:
    """Map a molecular subgraph onto lattice coordinates deterministically."""
    transformed = nx.Graph()

    mapped_nodes = sorted(set(mapping.values()))
    transformed.add_nodes_from(mapped_nodes)

    for u, v in current_graph.edges():
        if u not in mapping or v not in mapping:
            continue
        mu = mapping[u]
        mv = mapping[v]
        if mu == mv:
            continue

        # Keep only edges allowed by lattice when both nodes exist there.
        if lattice.number_of_nodes() == 0:
            transformed.add_edge(mu, mv)
        elif lattice.has_node(mu) and lattice.has_node(mv):
            if lattice.number_of_edges() == 0 or lattice.has_edge(mu, mv):
                transformed.add_edge(mu, mv)

    new_state = state.model_copy(update={
        "graph": transformed,
        "lattice": lattice,
    })
    return transformed, new_state


@register_atom(witness_quantum_mwis_solver)
@icontract.require(lambda mis_sample_quantity: mis_sample_quantity > 0, "mis_sample_quantity must be positive")
def quantum_mwis_solver(
    graph: nx.Graph,
    lattice_id_coord_dic: dict[int, tuple[float, float]],
    mis_sample_quantity: int,
    state: MolecularDockingState,
) -> tuple[list[set[int]], MolecularDockingState]:
    """Deterministic MWIS heuristic used as a quantum-solver placeholder."""
    # Greedy independent set by (degree, node_id).
    ordered_nodes = sorted(graph.nodes(), key=lambda n: (graph.degree[n], n))

    selected: set[int] = set()
    blocked: set[int] = set()
    for node in ordered_nodes:
        if node in blocked:
            continue
        selected.add(int(node))
        blocked.add(node)
        blocked.update(graph.neighbors(node))

    solutions: List[Set[int]] = [set(selected) for _ in range(mis_sample_quantity)]

    new_state = state.model_copy(update={
        "graph": graph,
        "lattice_id_coord_dic": lattice_id_coord_dic,
    })
    return solutions, new_state
