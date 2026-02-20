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
    """Extract deterministic embeddable node mappings from a graph.

    <!-- conceptual_profile
    {
        "abstract_name": "Topological Sub-Structure Mapper",
        "conceptual_transform": "Generates a set of potential mappings from a global relational graph to a smaller, localized sub-structure. It decomposes a complex connectivity map into a list of candidate embeddings for downstream processing.",
        "abstract_inputs": [
            {
                "name": "current_graph",
                "description": "A relational graph representing the global connectivity of a system."
            },
            {
                "name": "subgraph_quantity",
                "description": "An integer specifying the maximum number of sub-mappings to generate."
            },
            {
                "name": "state",
                "description": "A state object tracking the current topological configuration."
            }
        ],
        "abstract_outputs": [
            {
                "name": "mappings",
                "description": "A list of dictionaries, each mapping source elements to target sub-indices."
            },
            {
                "name": "new_state",
                "description": "The updated topological state object."
            }
        ],
        "algorithmic_properties": [
            "graph-decomposition",
            "embedding-generation",
            "deterministic-mapping"
        ],
        "cross_disciplinary_applications": [
            "Identifying common functional motifs in a social interaction network.",
            "Finding repeating structural patterns in a large-scale integrated circuit design.",
            "Extracting localized sub-clusters from a global knowledge graph for focused analysis."
        ]
    }
    /conceptual_profile -->
    """
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
    """Map a subgraph onto lattice coordinates deterministically.

    <!-- conceptual_profile
    {
        "abstract_name": "Lattice-Constrained Coordinate Projector",
        "conceptual_transform": "Projects the connectivity of a source graph onto a target coordinate system (lattice) using a specified node mapping. It ensures that only relationships supported by the target lattice are preserved in the transformed state, effectively filtering the graph through a spatial constraint.",
        "abstract_inputs": [
            {
                "name": "current_graph",
                "description": "The source relational graph to be projected."
            },
            {
                "name": "lattice",
                "description": "A target graph representing the allowable connections in the destination coordinate system."
            },
            {
                "name": "mapping",
                "description": "A dictionary defining the correspondence between source elements and target locations."
            },
            {
                "name": "state",
                "description": "A state object tracking the transformation context."
            }
        ],
        "abstract_outputs": [
            {
                "name": "transformed",
                "description": "A new relational graph representing the source connectivity as constrained by the target lattice."
            },
            {
                "name": "new_state",
                "description": "The updated transformation state object."
            }
        ],
        "algorithmic_properties": [
            "coordinate-projection",
            "subgraph-filtering",
            "relational-alignment"
        ],
        "cross_disciplinary_applications": [
            "Mapping a software dependency graph onto a fixed hardware processor topology.",
            "Projecting a logical network architecture onto a physical infrastructure layout.",
            "Aligning a theoretical relational structure to a rigid crystalline lattice for simulation."
        ]
    }
    /conceptual_profile -->
    """
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
    """Deterministic MWIS heuristic used as a combinatorial optimization solver placeholder.

    <!-- conceptual_profile
    {
        "abstract_name": "Global Conflict-Free Subset Optimizer",
        "conceptual_transform": "Identifies the largest possible subset of elements in a relational graph such that no two selected elements share a direct connection (conflict). It solves a global combinatorial optimization problem where selection is constrained by local interference.",
        "abstract_inputs": [
            {
                "name": "graph",
                "description": "A relational graph where edges represent conflicts or interference between elements."
            },
            {
                "name": "lattice_id_coord_dic",
                "description": "A dictionary mapping element identifiers to their physical coordinates."
            },
            {
                "name": "mis_sample_quantity",
                "description": "An integer specifying the number of optimal or near-optimal subsets to return."
            },
            {
                "name": "state",
                "description": "A state object tracking the optimization context."
            }
        ],
        "abstract_outputs": [
            {
                "name": "solutions",
                "description": "A list of sets, each containing a conflict-free subset of elements."
            },
            {
                "name": "new_state",
                "description": "The updated optimization state object."
            }
        ],
        "algorithmic_properties": [
            "combinatorial-optimization",
            "independent-set-solver",
            "conflict-resolution",
            "heuristic-driven"
        ],
        "cross_disciplinary_applications": [
            "Allocating non-interfering frequencies to neighboring towers in a cellular network.",
            "Scheduling independent tasks on a shared resource where certain tasks cannot run simultaneously.",
            "Finding the maximum set of non-overlapping regions for spatial partitioning in geographic information systems (GIS)."
        ]
    }
    /conceptual_profile -->
    """
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
