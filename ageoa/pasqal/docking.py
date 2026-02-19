from typing import Any, Tuple, List, Dict, Set
import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from state_models import MolecularDockingState
from witnesses import witness_sub_graph_embedder, witness_graph_transformer, witness_quantum_mwis_solver

class greedy_subgraph_solver_vv:
    graph: Any
    lattice: Any
    lattice_id_coord_dic: Any
    mappings: Any
    subgraph: Any
    solution: Any
    def obtain_embeddable_subgraphs(self, g: Any, q: int) -> None: ...
    def generate_graph_to_solve(self, g: Any, l: Any, m: Any) -> None: ...
    def mis_solving_function(self, g: Any, d: Any, q: int) -> None: ...

@register_atom(witness_sub_graph_embedder)
def sub_graph_embedder(current_graph: nx.Graph, subgraph_quantity: int, state: MolecularDockingState) -> tuple[list[dict[int, int]], MolecularDockingState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Extracts embeddable subgraphs from the main molecule graph using greedy lattice mapping.
    """
    obj = greedy_subgraph_solver_vv.__new__(greedy_subgraph_solver_vv)
    obj.graph = state.graph
    obj.lattice = state.lattice
    obj.lattice_id_coord_dic = state.lattice_id_coord_dic
    obj.obtain_embeddable_subgraphs(current_graph, subgraph_quantity)
    new_state = state.model_copy(update={
        "graph": obj.graph,
        "lattice": obj.lattice,
        "lattice_id_coord_dic": obj.lattice_id_coord_dic,
    })
    result = obj.mappings
    return result, new_state

@register_atom(witness_graph_transformer)
def graph_transformer(current_graph: nx.Graph, lattice: nx.Graph, mapping: dict[int, int], state: MolecularDockingState) -> tuple[nx.Graph, MolecularDockingState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Converts a molecular subgraph mapping into a solveable graph on the quantum lattice.
    """
    obj = greedy_subgraph_solver_vv.__new__(greedy_subgraph_solver_vv)
    obj.graph = state.graph
    obj.lattice = state.lattice
    obj.lattice_id_coord_dic = state.lattice_id_coord_dic
    obj.generate_graph_to_solve(current_graph, lattice, mapping)
    new_state = state.model_copy(update={
        "graph": obj.graph,
        "lattice": obj.lattice,
        "lattice_id_coord_dic": obj.lattice_id_coord_dic,
    })
    result = obj.subgraph
    return result, new_state

@register_atom(witness_quantum_mwis_solver)
def quantum_mwis_solver(graph: nx.Graph, lattice_id_coord_dic: dict[int, tuple[float, float]], mis_sample_quantity: int, state: MolecularDockingState) -> tuple[list[set[int]], MolecularDockingState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Solves the Maximum Weighted Independent Set problem using a quantum algorithm.
    """
    obj = greedy_subgraph_solver_vv.__new__(greedy_subgraph_solver_vv)
    obj.graph = state.graph
    obj.lattice = state.lattice
    obj.lattice_id_coord_dic = state.lattice_id_coord_dic
    obj.mis_solving_function(graph, lattice_id_coord_dic, mis_sample_quantity)
    new_state = state.model_copy(update={
        "graph": obj.graph,
        "lattice": obj.lattice,
        "lattice_id_coord_dic": obj.lattice_id_coord_dic,
    })
    result = obj.solution
    return result, new_state
