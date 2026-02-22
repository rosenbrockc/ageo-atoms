import networkx as nx
import pytest
import icontract

from ageoa.pasqal.docking import graph_transformer, quantum_mwis_solver, sub_graph_embedder
from ageoa.pasqal.docking_state import MolecularDockingState


def _build_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4)])
    return g


def _build_lattice() -> nx.Graph:
    l = nx.Graph()
    l.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
    return l


class TestPasqalPipeline:
    def test_runs_deterministically(self):
        graph = _build_graph()
        lattice = _build_lattice()
        state = MolecularDockingState(
            graph=graph,
            lattice=lattice,
            lattice_id_coord_dic={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)},
        )

        mappings, state = sub_graph_embedder(graph, 2, state)
        assert len(mappings) == 2
        assert mappings[0] == {1: 0}

        transformed, state = graph_transformer(graph, lattice, {1: 0, 2: 1, 3: 2, 4: 3}, state)
        assert isinstance(transformed, nx.Graph)
        assert transformed.number_of_nodes() > 0

        solutions, _ = quantum_mwis_solver(transformed, state.lattice_id_coord_dic or {}, 3, state)
        assert len(solutions) == 3
        assert all(isinstance(s, set) for s in solutions)
        assert solutions[0] == solutions[1] == solutions[2]


class TestPasqalPreconditions:
    def test_sub_graph_embedder_rejects_zero_k(self):
        graph = _build_graph()
        lattice = _build_lattice()
        state = MolecularDockingState(graph=graph, lattice=lattice, lattice_id_coord_dic={})

        with pytest.raises(icontract.ViolationError):
            sub_graph_embedder(graph, 0, state)

    def test_quantum_mwis_rejects_zero_samples(self):
        graph = _build_graph()
        lattice = _build_lattice()
        state = MolecularDockingState(graph=graph, lattice=lattice, lattice_id_coord_dic={})

        with pytest.raises(icontract.ViolationError):
            quantum_mwis_solver(graph, {}, 0, state)
