"""Molecular docking family runtime probe plans split from the monolithic registry."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_permutation_list = rt._assert_permutation_list
    _assert_scalar = rt._assert_scalar
    _assert_search_space = rt._assert_search_space
    _assert_state_snapshot = rt._assert_state_snapshot
    _assert_value = rt._assert_value

    def _molecular_docking_plans() -> dict[str, ProbePlan]:
        def _add_quantum_link_positive(func: Callable[..., Any]) -> Any:
            import networkx as nx

            graph = nx.Graph()
            graph.add_nodes_from(["A", "B"])
            return func(graph, "A", "B", 3)

        def _assert_add_quantum_link_result(result: Any) -> None:
            import networkx as nx

            assert isinstance(result, nx.Graph)
            assert result.has_edge("A", "_qlink_A_B_0")
            assert result.has_edge("_qlink_A_B_0", "_qlink_A_B_1")
            assert result.has_edge("_qlink_A_B_1", "B")

        def _assert_permutation_rows(result: Any) -> None:
            arr = np.asarray(result)
            assert arr.ndim == 2
            width = arr.shape[1]
            expected = np.arange(width)
            for row in arr:
                np.testing.assert_array_equal(np.sort(row), expected)

        def _greedy_mapping_context() -> tuple[Any, Any]:
            import networkx as nx

            graph = nx.Graph()
            graph.add_edges_from([(0, 1), (1, 2)])
            lattice = nx.Graph()
            lattice.add_edges_from([(10, 11), (11, 12)])
            return graph, lattice

        def _assert_mapping_context(result: Any) -> None:
            import networkx as nx

            assert isinstance(result, dict)
            assert isinstance(result["graph"], nx.Graph)
            assert isinstance(result["lattice"], nx.Graph)
            assert isinstance(result["lattice_instance"], nx.Graph)
            assert result["previously_generated_subgraphs"] == []
            assert result["seed"] == 7

        def _assert_initialized_frontier(result: Any) -> None:
            assert isinstance(result, dict)
            assert result["mapping"] == {0: 10}
            assert result["unmapping"] == {10: 0}
            assert result["unexpanded_nodes"] == {0}

        def _assert_greedy_extension(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            state, scores = result
            assert isinstance(state, dict)
            assert isinstance(scores, dict)
            assert state["mapping"] == {0: 10, 1: 11}
            assert state["unmapping"] == {10: 0, 11: 1}
            assert state["unexpanded_nodes"] == {0, 1}
            assert scores == {1: 1.0, 2: 0.0}

        def _assert_mapping_valid(result: Any) -> None:
            assert result is True

        def _assert_greedy_pipeline(result: Any) -> None:
            import networkx as nx

            assert isinstance(result, tuple) and len(result) == 2
            generated_subgraph, final_state = result
            assert isinstance(generated_subgraph, nx.Graph)
            assert set(generated_subgraph.nodes()) == {0, 1}
            assert set(generated_subgraph.edges()) == {(0, 1)}
            assert isinstance(final_state, dict)
            assert final_state["mapping"] == {0: 10, 1: 11}

        def _assert_greedy_mapping_d12_context(result: Any) -> None:
            import networkx as nx

            assert isinstance(result, dict)
            assert isinstance(result["graph"], nx.Graph)
            assert isinstance(result["lattice"], nx.Graph)
            assert isinstance(result["lattice_instance"], nx.Graph)
            assert result["seed"] == 11
            assert isinstance(result["previously_generated_subgraphs"], list)

        def _assert_greedy_mapping_d12_state(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            state, scores = result
            assert isinstance(state, dict)
            assert set(state) == {"mapping", "unmapping", "unexpanded_nodes"}
            assert isinstance(scores, list)
            assert len(scores) >= 1
            for item in scores:
                assert set(item) == {"node", "score"}

        def _assert_greedy_mapping_d12_validation(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            mapping_state, is_valid = result
            assert isinstance(mapping_state, dict)
            assert set(mapping_state) == {"mapping", "unmapping", "unexpanded_nodes"}
            assert is_valid is True

        def _assert_bandwidth_proposal(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 4
            iteration_state, candidate_permutation, candidate_matrix, candidate_bandwidth = result
            assert isinstance(iteration_state, np.ndarray)
            assert iteration_state.shape == (1,)
            assert isinstance(candidate_permutation, list)
            assert sorted(candidate_permutation) == [0, 1, 2]
            assert isinstance(candidate_matrix, np.ndarray)
            assert candidate_matrix.shape == (3, 3)
            assert isinstance(candidate_bandwidth, int)
            assert candidate_bandwidth >= 0

        def _assert_bandwidth_state_update(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            next_state, continue_search = result
            assert isinstance(next_state, np.ndarray)
            assert next_state.shape == (1,)
            state = next_state[0]
            assert state["bandwidth"] == 1
            assert state["remaining_iterations"] == 99
            assert state["accumulated_permutation"] == [2, 1, 0]
            assert continue_search is True

        return {
            "ageoa.molecular_docking.greedy_mapping.assemblestaticmappingcontext": ProbePlan(
                positive=ProbeCase(
                    "assemble deterministic static mapping context from graph and lattice inputs",
                    lambda func: func(*_greedy_mapping_context(), [], 7),
                    _assert_mapping_context,
                ),
                negative=ProbeCase(
                    "reject a missing graph input",
                    lambda func: func(None, _greedy_mapping_context()[1], [], 7),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.greedy_mapping.initializefrontierfromstartnode": ProbePlan(
                positive=ProbeCase(
                    "seed a frontier by mapping the starting node to the first free lattice node",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 7,
                        },
                        0,
                        {},
                        {},
                        set(),
                    ),
                    _assert_initialized_frontier,
                ),
                negative=ProbeCase(
                    "reject a missing mapping context",
                    lambda func: func(None, 0, {}, {}, set()),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.greedy_mapping.scoreandextendgreedycandidates": ProbePlan(
                positive=ProbeCase(
                    "score candidate nodes by mapped-neighbor support and greedily extend one step",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 7,
                        },
                        [1, 2],
                        {0},
                        [11, 12],
                        {0: 10},
                        {10: 0},
                        True,
                        True,
                    ),
                    _assert_greedy_extension,
                ),
                negative=ProbeCase(
                    "reject a missing considered-node list",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 7,
                        },
                        None,
                        {0},
                        [11, 12],
                        {0: 10},
                        {10: 0},
                        True,
                        True,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.greedy_mapping.validatecurrentmapping": ProbePlan(
                positive=ProbeCase(
                    "validate a deterministic edge-preserving partial graph-lattice mapping",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 7,
                        },
                        {0: 10, 1: 11},
                        {10: 0, 11: 1},
                    ),
                    _assert_mapping_valid,
                ),
                negative=ProbeCase(
                    "reject an inconsistent inverse mapping",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 7,
                        },
                        {0: 10, 1: 11},
                        {10: 0, 11: 2},
                    ),
                    _assert_value(False),
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.greedy_mapping.rungreedymappingpipeline": ProbePlan(
                positive=ProbeCase(
                    "assemble the greedy-mapping pipeline output from a validated partial state",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 7,
                        },
                        0,
                        True,
                        True,
                        {"mapping": {0: 10}, "unmapping": {10: 0}, "unexpanded_nodes": {0}},
                        {"mapping": {0: 10, 1: 11}, "unmapping": {10: 0, 11: 1}, "unexpanded_nodes": {0, 1}},
                        True,
                    ),
                    _assert_greedy_pipeline,
                ),
                negative=ProbeCase(
                    "reject a missing mapping context in the orchestration stage",
                    lambda func: func(None, 0, True, True, {"mapping": {}}, {"mapping": {}}, True),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.greedy_mapping_d12.init_problem_context": ProbePlan(
                positive=ProbeCase(
                    "initialize a deterministic refined-ingest greedy-mapping context",
                    lambda func: func(
                        _greedy_mapping_context()[0],
                        _greedy_mapping_context()[1],
                        [],
                        11,
                    ),
                    _assert_greedy_mapping_d12_context,
                ),
                negative=ProbeCase(
                    "reject a missing graph in refined-ingest greedy-mapping context initialization",
                    lambda func: func(None, _greedy_mapping_context()[1], [], 11),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion": ProbePlan(
                positive=ProbeCase(
                    "construct one deterministic refined-ingest greedy mapping expansion state",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 11,
                        },
                        0,
                        {"mapping": {}, "unmapping": {}, "unexpanded_nodes": set()},
                        [0, 1, 2],
                        True,
                        True,
                    ),
                    _assert_greedy_mapping_d12_state,
                ),
                negative=ProbeCase(
                    "reject a missing problem context in refined-ingest greedy mapping expansion",
                    lambda func: func(None, 0, {"mapping": {}, "unmapping": {}, "unexpanded_nodes": set()}, [0, 1], True, True),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate": ProbePlan(
                positive=ProbeCase(
                    "validate one deterministic refined-ingest greedy mapping state",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 11,
                        },
                        0,
                        True,
                        True,
                        {"mapping": {0: 10, 1: 11}, "unmapping": {10: 0, 11: 1}, "unexpanded_nodes": {0, 1}},
                    ),
                    _assert_greedy_mapping_d12_validation,
                ),
                negative=ProbeCase(
                    "reject a missing mapping state in refined-ingest greedy mapping orchestration",
                    lambda func: func(
                        {
                            "graph": _greedy_mapping_context()[0],
                            "lattice": _greedy_mapping_context()[1],
                            "lattice_instance": _greedy_mapping_context()[1],
                            "previously_generated_subgraphs": [],
                            "seed": 11,
                        },
                        0,
                        True,
                        True,
                        None,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.quantum_mwis_solver": ProbePlan(
                positive=ProbeCase(
                    "MWIS solver falls back to a deterministic median threshold on 1D input",
                    lambda func: func(np.array([1.0, 3.0, 2.0])),
                    _assert_array(np.array([0.0, 1.0, 1.0])),
                ),
                negative=ProbeCase(
                    "MWIS solver rejects empty input",
                    lambda func: func(np.array([])),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.add_quantum_link.addquantumlink": ProbePlan(
                positive=ProbeCase(
                    "quantum link insertion creates a deterministic chain between nodes",
                    _add_quantum_link_positive,
                    _assert_add_quantum_link_result,
                ),
                negative=ProbeCase(
                    "quantum link insertion rejects a missing graph",
                    lambda func: func(None, "A", "B", 2),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.mwis_sa.to_qubo": ProbePlan(
                positive=ProbeCase(
                    "MWIS-to-QUBO conversion maps diagonal weights and edge penalties",
                    lambda func: func(np.array([[2.0, 1.0], [1.0, 3.0]]), 5.0),
                    _assert_array(np.array([[-2.0, 5.0], [5.0, -3.0]])),
                ),
                negative=ProbeCase(
                    "MWIS-to-QUBO rejects a missing graph",
                    lambda func: func(None, 5.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.enumerate_threshold_based_permutations": ProbePlan(
                positive=ProbeCase(
                    "threshold-based permutation enumeration returns valid permutations",
                    lambda func: func(
                        np.array(
                            [
                                [0.0, 2.0, 0.0],
                                [2.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0],
                            ]
                        ),
                        2.0,
                        np.array([0.25, 0.75]),
                    ),
                    _assert_permutation_rows,
                ),
                negative=ProbeCase(
                    "threshold permutation enumeration rejects a non-float amplitude",
                    lambda func: func(np.eye(2), "bad", np.array([0.25])),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.validate_square_matrix_shape": ProbePlan(
                positive=ProbeCase(
                    "square-matrix validation passes through a 3x3 matrix",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                ),
                negative=ProbeCase(
                    "square-matrix validation rejects a rectangular matrix",
                    lambda func: func(np.array([[1.0, 2.0, 3.0]])),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.compute_absolute_weighted_index_distances": ProbePlan(
                positive=ProbeCase(
                    "weighted index distance calculation matches elementwise |value * (i-j)|",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                ),
                negative=ProbeCase(
                    "weighted distance calculation rejects a missing matrix",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.aggregate_maximum_distance_as_bandwidth": ProbePlan(
                positive=ProbeCase(
                    "bandwidth aggregation returns the maximum weighted distance",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_scalar(2.0),
                ),
                negative=ProbeCase(
                    "bandwidth aggregation rejects missing distances",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.validate_symmetric_input": ProbePlan(
                positive=ProbeCase(
                    "symmetric-input validation passes through a symmetric matrix",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                ),
                negative=ProbeCase(
                    "symmetric-input validation rejects an asymmetric matrix",
                    lambda func: func(np.array([[0.0, 1.0], [0.0, 0.0]])),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.initialize_reduction_state": ProbePlan(
                positive=ProbeCase(
                    "reduction-state initialization creates a working matrix snapshot",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_state_snapshot(1, 100),
                ),
                negative=ProbeCase(
                    "reduction-state initialization rejects a missing matrix",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.enforce_threshold_sparsity": ProbePlan(
                positive=ProbeCase(
                    "threshold sparsity zeros entries below the threshold",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), 1.5),
                    _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
                ),
                negative=ProbeCase(
                    "threshold sparsity rejects a non-float threshold",
                    lambda func: func(np.eye(2), "bad"),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.build_sparse_graph_view": ProbePlan(
                positive=ProbeCase(
                    "sparse-graph view preserves the thresholded matrix content",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                ),
                negative=ProbeCase(
                    "sparse-graph view rejects a missing matrix",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.compute_symmetric_bandwidth_reducing_order": ProbePlan(
                positive=ProbeCase(
                    "RCM ordering produces a valid reverse bandwidth permutation",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_array(np.array([2, 1, 0])),
                ),
                negative=ProbeCase(
                    "RCM ordering rejects a missing sparse matrix",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.build_threshold_search_space": ProbePlan(
                positive=ProbeCase(
                    "threshold search space returns matrix amplitude and the 0.1..0.99 sweep",
                    lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                    _assert_search_space(2.0, 90),
                ),
                negative=ProbeCase(
                    "threshold search space rejects a missing matrix",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.select_minimum_bandwidth_permutation": ProbePlan(
                positive=ProbeCase(
                    "minimum-bandwidth selector returns the best candidate permutation",
                    lambda func: func(
                        np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                        np.array([[0, 1, 2], [2, 1, 0]]),
                    ),
                    _assert_permutation_list([0, 1, 2]),
                ),
                negative=ProbeCase(
                    "minimum-bandwidth selector rejects missing candidates",
                    lambda func: func(np.eye(2), None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.extract_final_permutation": ProbePlan(
                positive=ProbeCase(
                    "final-permutation extraction returns the accumulated permutation",
                    lambda func: func(
                        np.array(
                            [
                                {
                                    "working_matrix": np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                                    "accumulated_permutation": [0, 1, 2],
                                    "bandwidth": 1,
                                    "remaining_iterations": 100,
                                }
                            ],
                            dtype=object,
                        )
                    ),
                    _assert_permutation_list([0, 1, 2]),
                ),
                negative=ProbeCase(
                    "final-permutation extraction rejects a missing terminal state",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.propose_greedy_permutation_step": ProbePlan(
                positive=ProbeCase(
                    "propose one greedy reverse-Cuthill-McKee permutation step",
                    lambda func: func(
                        np.array(
                            [
                                {
                                    "working_matrix": np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                                    "accumulated_permutation": [0, 1, 2],
                                    "bandwidth": 2,
                                    "remaining_iterations": 100,
                                }
                            ],
                            dtype=object,
                        )
                    ),
                    _assert_bandwidth_proposal,
                ),
                negative=ProbeCase(
                    "reject a missing reduction state when proposing a greedy permutation step",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.minimize_bandwidth.update_state_with_improvement_criterion": ProbePlan(
                positive=ProbeCase(
                    "accept an improved bandwidth candidate and decrement remaining iterations",
                    lambda func: func(
                        np.array(
                            [
                                {
                                    "working_matrix": np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                                    "accumulated_permutation": [0, 1, 2],
                                    "bandwidth": 2,
                                    "remaining_iterations": 100,
                                }
                            ],
                            dtype=object,
                        ),
                        [2, 1, 0],
                        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]]),
                        1,
                    ),
                    _assert_bandwidth_state_update,
                ),
                negative=ProbeCase(
                    "reject a missing candidate matrix during bandwidth-state update",
                    lambda func: func(
                        np.array(
                            [
                                {
                                    "working_matrix": np.eye(2),
                                    "accumulated_permutation": [0, 1],
                                    "bandwidth": 1,
                                    "remaining_iterations": 10,
                                }
                            ],
                            dtype=object,
                        ),
                        [0, 1],
                        None,
                        1,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
        }


    def _molecular_docking_quantum_solver_plans() -> dict[str, ProbePlan]:
        def _assert_quantum_problem_bundle(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 5
            register, parameters, permutation_list, backend_flags, num_sol = result
            assert register == {0: (0.0, 0.0), 1: (1.0, 0.0)}
            assert parameters["duration"] == 4000.0
            assert parameters["detuning_maximum"] == 5.0
            assert parameters["amplitude_maximum"] == 5.0
            assert parameters["register"] == register
            assert permutation_list == [[0, 1]]
            assert backend_flags == {"run_qutip": False, "run_emu_mps": False, "run_sv": True}
            assert num_sol == 2

        def _assert_quantum_sample_bundle(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            measurement_counts, final_register = result
            assert isinstance(measurement_counts, dict)
            assert measurement_counts
            assert set(measurement_counts.keys()).issubset({"00", "01", "10", "11"})
            assert sum(int(v) for v in measurement_counts.values()) == 500
            assert final_register == {0: (0.0, 0.0), 1: (1.0, 0.0)}

        def _assert_solution_list(expected: list[list[int]]) -> Callable[[Any], None]:
            def _assert(result: Any) -> None:
                assert isinstance(result, list)
                assert result == expected

            return _assert

        def _quantum_solver_graph() -> Any:
            import networkx as nx

            graph = nx.Graph()
            graph.add_node(0, weight=1.0)
            graph.add_node(1, weight=2.0)
            graph.add_edge(0, 1)
            return graph

        return {
            "ageoa.molecular_docking.quantum_solver.quantumproblemdefinition": ProbePlan(
                positive=ProbeCase(
                    "prepare a deterministic two-node quantum problem definition bundle",
                    lambda func: func(
                        _quantum_solver_graph(),
                        {0: (0.0, 0.0), 1: (1.0, 0.0)},
                        2,
                        False,
                    ),
                    _assert_quantum_problem_bundle,
                ),
                negative=ProbeCase(
                    "reject a missing problem graph",
                    lambda func: func(None, {0: (0.0, 0.0), 1: (1.0, 0.0)}, 2, False),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.quantum_solver.adiabaticquantumsampler": ProbePlan(
                positive=ProbeCase(
                    "sample a deterministic two-node quantum register with a fixed RNG path",
                    lambda func: func(
                        {0: (0.0, 0.0), 1: (1.0, 0.0)},
                        {
                            "graph": _quantum_solver_graph(),
                            "duration": 4000.0,
                            "detuning_maximum": 5.0,
                            "amplitude_maximum": 5.0,
                        },
                        [[0, 1]],
                        {"run_qutip": False, "run_emu_mps": False, "run_sv": True},
                    ),
                    _assert_quantum_sample_bundle,
                ),
                negative=ProbeCase(
                    "reject a missing register definition",
                    lambda func: func(
                        None,
                        {
                            "graph": _quantum_solver_graph(),
                            "duration": 4000.0,
                            "detuning_maximum": 5.0,
                            "amplitude_maximum": 5.0,
                        },
                        [[0, 1]],
                        {"run_qutip": False, "run_emu_mps": False, "run_sv": True},
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.molecular_docking.quantum_solver.solutionextraction": ProbePlan(
                positive=ProbeCase(
                    "extract the two most frequent bitstring solutions from a deterministic count map",
                    lambda func: func(
                        {"10": 7, "01": 3},
                        {0: (0.0, 0.0), 1: (1.0, 0.0)},
                        2,
                    ),
                    _assert_solution_list([[0], [1]]),
                ),
                negative=ProbeCase(
                    "reject a missing measurement count distribution",
                    lambda func: func(None, {0: (0.0, 0.0), 1: (1.0, 0.0)}, 2),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
        }


    plans: dict[str, Any] = {}
    plans.update(_molecular_docking_plans())
    plans.update(_molecular_docking_quantum_solver_plans())
    return plans
