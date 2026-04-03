"""Runtime probe plans for the mixed hftbacktest and ingest-heavy family block."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.spatial as spatial


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_batch_plan = rt._assert_batch_plan
    _assert_dataset_state = rt._assert_dataset_state
    _assert_dict_keys = rt._assert_dict_keys
    _assert_draw_bundle = rt._assert_draw_bundle
    _assert_float_int_pair = rt._assert_float_int_pair
    _assert_inventory_adjusted_quotes = rt._assert_inventory_adjusted_quotes
    _assert_market_maker_state = rt._assert_market_maker_state
    _assert_nonincreasing_float_list = rt._assert_nonincreasing_float_list
    _assert_positive_weights = rt._assert_positive_weights
    _assert_quantum_solution_extractor = rt._assert_quantum_solution_extractor
    _assert_quantum_solver_orchestrator_result = rt._assert_quantum_solver_orchestrator_result
    _assert_profitable_cycles = rt._assert_profitable_cycles
    _assert_scalar = rt._assert_scalar
    _assert_shape = rt._assert_shape
    _assert_tuple = rt._assert_tuple
    _assert_type = rt._assert_type

    class _DummyBlock:
        def __init__(self, value: int) -> None:
            self.value = value

    return {
        "ageoa.hftbacktest.initialize_glft_state": ProbePlan(
            positive=ProbeCase(
                "initialize GLFT state returns zero coefficients",
                lambda func: func(),
                _assert_tuple((0.0, 0.0)),
            ),
            parity_used=True,
        ),
        "ageoa.hftbacktest.update_glft_coefficients": ProbePlan(
            positive=ProbeCase(
                "GLFT coefficient update on a simple numeric input",
                lambda func: func(0.0, 0.0, 2.0, 0.1, 1.0, 2.0, 4.0),
                _assert_tuple((0.05, 0.27465307216702745)),
            ),
            negative=ProbeCase(
                "GLFT update rejects non-numeric xi",
                lambda func: func(0.0, 0.0, "bad", 0.1, 1.0, 2.0, 4.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.hftbacktest.evaluate_spread_conditions": ProbePlan(
            positive=ProbeCase(
                "spread evaluation computes a half-spread and validity flag",
                lambda func: func(2.0, 0.5, 1.5, 2.0, 0.25, 1.0),
                _assert_tuple((1.25, True)),
            ),
            negative=ProbeCase(
                "spread evaluation rejects non-numeric c1",
                lambda func: func("bad", 0.5, 1.5, 2.0, 0.25, 1.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.market_making_avellaneda": ProbePlan(
            positive=ProbeCase(
                "Avellaneda-Stoikov spread series over a small price vector",
                lambda func: func(np.array([100.0, 101.0, 102.0])),
                _assert_array(np.array([1.2907704244963784, 1.2907704244963784, 1.2907704244963784])),
            ),
            negative=ProbeCase(
                "Avellaneda-Stoikov rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.almgren_chriss_execution": ProbePlan(
            positive=ProbeCase(
                "Almgren-Chriss execution returns a linear liquidation trajectory",
                lambda func: func(np.array([12.0, 0.0, 0.0, 0.0])),
                _assert_array(np.array([12.0, 9.0, 6.0, 3.0])),
            ),
            negative=ProbeCase(
                "Almgren-Chriss rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.almgren_chriss_v2.riskaversioninit": ProbePlan(
            positive=ProbeCase(
                "refined-ingest Almgren-Chriss risk aversion bootstrap preserves the scalar parameter",
                lambda func: func(0.5),
                _assert_scalar(0.5),
            ),
            negative=ProbeCase(
                "refined-ingest Almgren-Chriss bootstrap rejects a non-numeric risk aversion",
                lambda func: func("bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.almgren_chriss_v2.optimalexecutiontrajectory": ProbePlan(
            positive=ProbeCase(
                "refined-ingest Almgren-Chriss trajectory produces a nonincreasing liquidation schedule",
                lambda func: func(0.5, 100.0, 4),
                _assert_nonincreasing_float_list(0.0),
            ),
            negative=ProbeCase(
                "refined-ingest Almgren-Chriss trajectory rejects a non-numeric total share count",
                lambda func: func(0.5, "bad", 4),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.almgren_chriss.computeoptimaltrajectory": ProbePlan(
            positive=ProbeCase(
                "generated Almgren-Chriss trajectory returns a linear liquidation array",
                lambda func: func(12.0, 4, 0.5),
                _assert_array(np.array([12.0, 9.0, 6.0, 3.0, 0.0], dtype=float)),
            ),
            negative=ProbeCase(
                "generated Almgren-Chriss trajectory rejects a non-integer day count",
                lambda func: func(12.0, "bad", 0.5),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.order_flow_imbalance.orderflowimbalanceevaluation": ProbePlan(
            positive=ProbeCase(
                "generated OFI wrapper returns the expected bid-ask imbalance for two snapshots",
                lambda func: func(
                    {"bid_price": 101.0, "bid_size": 7.0, "ask_price": 103.0, "ask_size": 6.0},
                    {"bid_price": 100.0, "bid_size": 4.0, "ask_price": 104.0, "ask_size": 5.0},
                ),
                _assert_scalar(1.0),
            ),
            negative=ProbeCase(
                "generated OFI wrapper rejects a missing previous row",
                lambda func: func({"bid_price": 101.0, "bid_size": 7.0}, None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.pin_model.pinlikelihoodevaluation": ProbePlan(
            positive=ProbeCase(
                "generated PIN likelihood wrapper returns the expected squared-error score",
                lambda func: func([0.5, 0.25, 4.0, 1.0], np.array([1.0, 2.0]), np.array([2.0, 1.0])),
                _assert_scalar(3.0),
            ),
            negative=ProbeCase(
                "generated PIN likelihood wrapper rejects a missing parameter vector",
                lambda func: func(None, np.array([1.0, 2.0]), np.array([2.0, 1.0])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.pin_informed_trading": ProbePlan(
            positive=ProbeCase(
                "PIN estimator computes order-flow imbalance",
                lambda func: func(np.array([10.0, 8.0, 3.0, 1.0])),
                _assert_array(np.array([0.6363636363636364])),
            ),
            negative=ProbeCase(
                "PIN estimator rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.limit_order_queue_estimator": ProbePlan(
            positive=ProbeCase(
                "queue estimator normalizes cumulative queue position",
                lambda func: func(np.array([2.0, 3.0, 5.0])),
                _assert_array(np.array([0.2, 0.5, 1.0])),
            ),
            negative=ProbeCase(
                "queue estimator rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.avellaneda_stoikov.initializemarketmakerstate": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic Avellaneda-Stoikov market-maker state",
                lambda func: func(100.0, 2.0),
                _assert_market_maker_state(),
            ),
            negative=ProbeCase(
                "reject a non-numeric inventory level",
                lambda func: func(100.0, "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.avellaneda_stoikov.computeinventoryadjustedquotes": ProbePlan(
            positive=ProbeCase(
                "compute inventory-adjusted Avellaneda-Stoikov quotes from a deterministic state",
                lambda func: func(
                    {
                        "s": 100.0,
                        "q": 2.0,
                        "sigma": 0.02,
                        "gamma": 0.1,
                        "k": 1.5,
                        "T": 1.0,
                        "t": 0.25,
                    }
                ),
                _assert_inventory_adjusted_quotes(),
            ),
            negative=ProbeCase(
                "reject a non-dict market-maker state",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.avellaneda_stoikov_d12.marketmakerstateinit": ProbePlan(
            positive=ProbeCase(
                "refined-ingest Avellaneda-Stoikov state init returns the expected scalar tuple",
                lambda func: func(100.0, 2.0),
                _assert_tuple((0.1, 1.5, 2.0, 100.0, 0.02)),
            ),
            negative=ProbeCase(
                "refined-ingest Avellaneda-Stoikov state init rejects a non-numeric mid-price",
                lambda func: func("bad", 2.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.avellaneda_stoikov_d12.optimalquotecalculation": ProbePlan(
            positive=ProbeCase(
                "refined-ingest Avellaneda-Stoikov quote calculation returns a consistent quote tuple",
                lambda func: func(0.1, 1.5, 2.0, 100.0, 0.02),
                _assert_tuple((99.35451478862429, 100.64532521137572, 99.99992, 1.2908104227514234)),
            ),
            negative=ProbeCase(
                "refined-ingest Avellaneda-Stoikov quote calculation rejects a non-numeric volatility",
                lambda func: func(0.1, 1.5, 2.0, 100.0, "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.spatial_v2.voronoitessellation": ProbePlan(
            positive=ProbeCase(
                "refined-ingest Voronoi tessellation returns a SciPy Voronoi object for a small 2D point set",
                lambda func: func(
                    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float),
                ),
                _assert_type(spatial.Voronoi),
            ),
            negative=ProbeCase(
                "refined-ingest Voronoi tessellation rejects a missing point set",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.spatial_v2.delaunaytriangulation": ProbePlan(
            positive=ProbeCase(
                "refined-ingest Delaunay triangulation returns a SciPy Delaunay object for a small 2D point set",
                lambda func: func(
                    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float),
                ),
                _assert_type(spatial.Delaunay),
            ),
            negative=ProbeCase(
                "refined-ingest Delaunay triangulation rejects a missing point set",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.sparse_graph_v2.singlesourceshortestpath": ProbePlan(
            positive=ProbeCase(
                "refined-ingest sparse-graph single-source shortest path returns the expected distance vector",
                lambda func: func(
                    np.array(
                        [
                            [0.0, 1.0, 4.0],
                            [1.0, 0.0, 2.0],
                            [4.0, 2.0, 0.0],
                        ],
                        dtype=float,
                    ),
                    indices=0,
                ),
                _assert_array(np.array([0.0, 1.0, 3.0], dtype=float)),
            ),
            negative=ProbeCase(
                "refined-ingest sparse-graph single-source shortest path rejects a non-numeric limit",
                lambda func: func(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float), limit="bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.sparse_graph_v2.allpairsshortestpath": ProbePlan(
            positive=ProbeCase(
                "refined-ingest sparse-graph all-pairs shortest path returns the expected distance matrix",
                lambda func: func(
                    np.array(
                        [
                            [0.0, 1.0, 4.0],
                            [1.0, 0.0, 2.0],
                            [4.0, 2.0, 0.0],
                        ],
                        dtype=float,
                    )
                ),
                _assert_array(
                    np.array(
                        [
                            [0.0, 1.0, 3.0],
                            [1.0, 0.0, 2.0],
                            [3.0, 2.0, 0.0],
                        ],
                        dtype=float,
                    )
                ),
            ),
            negative=ProbeCase(
                "refined-ingest sparse-graph all-pairs shortest path rejects a missing graph",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.sparse_graph_v2.minimumspanningtree": ProbePlan(
            positive=ProbeCase(
                "refined-ingest sparse-graph minimum spanning tree returns the expected weighted tree",
                lambda func: func(
                    np.array(
                        [
                            [0.0, 1.0, 4.0],
                            [1.0, 0.0, 2.0],
                            [4.0, 2.0, 0.0],
                        ],
                        dtype=float,
                    )
                ).toarray(),
                _assert_array(
                    np.array(
                        [
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 2.0],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=float,
                    )
                ),
            ),
            negative=ProbeCase(
                "refined-ingest sparse-graph minimum spanning tree rejects a missing graph",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.astroflow.dedispersionkernel": ProbePlan(
            positive=ProbeCase(
                "dedispersion kernel averages delayed channel contributions into a deterministic output grid",
                lambda func: func(
                    np.array(
                        [
                            [1.0, 10.0],
                            [2.0, 20.0],
                            [3.0, 30.0],
                            [4.0, 40.0],
                            [5.0, 50.0],
                        ],
                        dtype=float,
                    ),
                    np.array([[0, 1], [1, 0]], dtype=int),
                    2,
                    1,
                    3,
                    2,
                    0,
                    32,
                ),
                _assert_array(np.array([[10.5, 16.0, 21.5], [6.0, 11.5, 17.0]], dtype=float)),
            ),
            negative=ProbeCase(
                "dedispersion kernel rejects a missing input array",
                lambda func: func(None, np.array([[0, 1], [1, 0]], dtype=int), 2, 1, 3, 2, 0, 32),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.build_interaction_graph.pair_distance_compatibility_check": ProbePlan(
            positive=ProbeCase(
                "pair-distance compatibility accepts a right-side distance inside the expanded left interval",
                lambda func: func(np.array([1.0, 3.0], dtype=float), np.array([2.4, 5.0], dtype=float), 0.5),
                _assert_scalar(True),
            ),
            negative=ProbeCase(
                "pair-distance compatibility rejects a non-numeric interaction threshold",
                lambda func: func(np.array([1.0, 3.0], dtype=float), np.array([2.4, 5.0], dtype=float), "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.build_interaction_graph.networkx_weighted_graph_materialization": ProbePlan(
            positive=ProbeCase(
                "weighted graph materialization returns a NetworkX graph with the provided weighted edge",
                lambda func: func([("a", "b", 1.5)], {"a", "b"}),
                _assert_type(__import__("networkx").Graph),
            ),
            negative=ProbeCase(
                "weighted graph materialization rejects missing edges",
                lambda func: func(None, {"a", "b"}),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_solver_d12.quantumsolverorchestrator": ProbePlan(
            positive=ProbeCase(
                "refined-ingest quantum solver orchestrator returns non-empty MWIS solutions and counts",
                lambda func: func(
                    __import__("networkx").Graph([(0, 1), (1, 2)]),
                    {
                        0: np.array([0.0, 0.0], dtype=float),
                        1: np.array([1.0, 0.0], dtype=float),
                        2: np.array([0.0, 1.0], dtype=float),
                    },
                    2,
                    False,
                ),
                _assert_quantum_solver_orchestrator_result(2),
            ),
            negative=ProbeCase(
                "refined-ingest quantum solver orchestrator rejects a non-graph input",
                lambda func: func([], {}, 2, False),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_solver_d12.interactionboundscomputer": ProbePlan(
            positive=ProbeCase(
                "refined-ingest interaction bounds computer returns a valid positive bound pair",
                lambda func: func(
                    {
                        0: np.array([0.0, 0.0], dtype=float),
                        1: np.array([1.0, 0.0], dtype=float),
                        2: np.array([0.0, 2.0], dtype=float),
                    },
                    __import__("networkx").Graph([(0, 1)]),
                ),
                _assert_float_int_pair(862690.0, 862690),
            ),
            negative=ProbeCase(
                "refined-ingest interaction bounds computer rejects an empty register layout",
                lambda func: func({}, __import__("networkx").Graph()),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_solver_d12.adiabaticpulseassembler": ProbePlan(
            positive=ProbeCase(
                "refined-ingest adiabatic pulse assembler returns a pulse-sequence dict",
                lambda func: func({0: np.array([0.0, 0.0], dtype=float)}, {"duration": 4000.0, "detuning_maximum": 2.0}),
                _assert_dict_keys({"register", "parameters", "duration", "type"}),
            ),
            negative=ProbeCase(
                "refined-ingest adiabatic pulse assembler rejects parameters without duration",
                lambda func: func({0: np.array([0.0, 0.0], dtype=float)}, {"detuning_maximum": 2.0}),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_solver_d12.quantumcircuitsampler": ProbePlan(
            positive=ProbeCase(
                "refined-ingest quantum circuit sampler returns a non-empty count dictionary",
                lambda func: func(
                    {"graph": __import__("networkx").Graph([(0, 1)])},
                    {0: np.array([0.0, 0.0], dtype=float), 1: np.array([1.0, 0.0], dtype=float)},
                    [0, 1],
                    False,
                    False,
                    True,
                ),
                _assert_type(dict),
            ),
            negative=ProbeCase(
                "refined-ingest quantum circuit sampler rejects a non-list permutation",
                lambda func: func({"graph": __import__("networkx").Graph([(0, 1)])}, {0: np.array([0.0, 0.0], dtype=float)}, None, False, False, True),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_solver_d12.quantumsolutionextractor": ProbePlan(
            positive=ProbeCase(
                "refined-ingest quantum solution extractor decodes top bitstrings into ranked solutions",
                lambda func: func({"10": 7, "01": 5, "11": 1}, {0: np.array([0.0, 0.0], dtype=float), 1: np.array([1.0, 0.0], dtype=float)}, 2),
                _assert_quantum_solution_extractor(2),
            ),
            negative=ProbeCase(
                "refined-ingest quantum solution extractor rejects an empty count distribution",
                lambda func: func({}, {0: np.array([0.0, 0.0], dtype=float)}, 2),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.hierarchical_risk_parity.compute_hrp_weights": ProbePlan(
            positive=ProbeCase(
                "compute HRP weights for a deterministic three-asset return matrix",
                lambda func: func(
                    np.array(
                        [
                            [0.01, 0.02, -0.01],
                            [0.00, 0.01, -0.02],
                            [0.02, 0.015, -0.005],
                            [0.01, 0.005, -0.01],
                        ],
                        dtype=float,
                    )
                ),
                _assert_positive_weights(3),
            ),
            negative=ProbeCase(
                "reject a one-dimensional return series",
                lambda func: func(np.array([0.01, 0.02, -0.01], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.hierarchical_risk_parity.hrppipelinerun": ProbePlan(
            positive=ProbeCase(
                "execute the HRP pipeline over a deterministic pandas DataFrame",
                lambda func: func(
                    __import__("pandas").DataFrame(
                        {
                            "asset_a": [0.01, 0.00, 0.02, 0.01],
                            "asset_b": [0.02, 0.01, 0.015, 0.005],
                            "asset_c": [-0.01, -0.02, -0.005, -0.01],
                        }
                    )
                ),
                _assert_positive_weights(3),
            ),
            negative=ProbeCase(
                "reject a single-asset DataFrame",
                lambda func: func(__import__("pandas").DataFrame({"only_asset": [0.01, 0.0, 0.02]})),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pulsar_folding.dm_can_brute_force": ProbePlan(
            positive=ProbeCase(
                "run deterministic brute-force DM search on a short folded profile",
                lambda func: func(np.array([0.0, 2.0, 1.0, 0.5], dtype=float)),
                _assert_array(np.array([0.0, 2.0, 1.0, 0.5], dtype=float)),
            ),
            negative=ProbeCase(
                "reject an empty folded profile",
                lambda func: func(np.array([], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pulsar_folding.spline_bandpass_correction": ProbePlan(
            positive=ProbeCase(
                "subtract a smooth spline baseline from a short bandpass trace",
                lambda func: func(np.array([1.0, 1.5, 2.0, 1.5, 1.0], dtype=float)),
                _assert_shape((5,)),
            ),
            negative=ProbeCase(
                "reject an empty bandpass trace",
                lambda func: func(np.array([], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.fractional_diff.fractional_differentiator": ProbePlan(
            positive=ProbeCase(
                "compute fractional differentiation on a short deterministic price series",
                lambda func: func(
                    __import__("pandas").Series([100.0, 101.0, 103.0, 102.0, 104.0]),
                    0.4,
                    0.01,
                ),
                _assert_type(__import__("pandas").Series),
            ),
            negative=ProbeCase(
                "reject a non-numeric differentiation order",
                lambda func: func(__import__("pandas").Series([1.0, 2.0, 3.0]), "bad", 0.01),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.encoding_dist_mat.encodedistancematrix": ProbePlan(
            positive=ProbeCase(
                "pad and stack a pair of small distance matrices into a batched tensor",
                lambda func: func(
                    np.array(
                        [
                            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                            np.array([[5.0]], dtype=float),
                        ],
                        dtype=object,
                    ),
                    2,
                    2,
                ),
                _assert_shape((2, 2, 2)),
            ),
            negative=ProbeCase(
                "reject a non-array matrix container",
                lambda func: func([np.array([[1.0]], dtype=float)], 1, 1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.fasta_dataset.dataset_state_initialization": ProbePlan(
            positive=ProbeCase(
                "build an in-memory FASTA dataset state from aligned labels and sequences",
                lambda func: func(["seq_a", "seq_b"], ["ACGT", "TT"]),
                _assert_dataset_state(["seq_a", "seq_b"], ["ACGT", "TT"]),
            ),
            negative=ProbeCase(
                "reject mismatched sequence label and sequence counts",
                lambda func: func(["seq_a"], ["ACGT", "TT"]),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.fasta_dataset.dataset_length_query": ProbePlan(
            positive=ProbeCase(
                "query the length of a tiny FASTA dataset state",
                lambda func: func({"sequence_labels": ["seq_a", "seq_b"], "sequence_strs": ["ACGT", "TT"]}),
                _assert_scalar(2),
            ),
            negative=ProbeCase(
                "reject a missing FASTA dataset state",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.fasta_dataset.dataset_item_retrieval": ProbePlan(
            positive=ProbeCase(
                "retrieve one labeled FASTA sequence by index",
                lambda func: func({"sequence_labels": ["seq_a", "seq_b"], "sequence_strs": ["ACGT", "TT"]}, 1),
                _assert_tuple(("seq_b", "TT")),
            ),
            negative=ProbeCase(
                "reject a non-integer FASTA dataset index",
                lambda func: func({"sequence_labels": ["seq_a"], "sequence_strs": ["ACGT"]}, "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.fasta_dataset.token_budget_batch_planning": ProbePlan(
            positive=ProbeCase(
                "plan one token-budget batch for a tiny FASTA dataset",
                lambda func: func(
                    {"sequence_labels": ["seq_a", "seq_b"], "sequence_strs": ["ACGT", "TT"]},
                    8,
                    1,
                ),
                _assert_batch_plan([[1], [0]]),
            ),
            negative=ProbeCase(
                "reject a non-integer token budget",
                lambda func: func({"sequence_labels": ["seq_a"], "sequence_strs": ["ACGT"]}, "bad", 1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.jax_advi.optimize_advi.posteriordrawsampling": ProbePlan(
            positive=ProbeCase(
                "draw constrained posterior samples from a simple mean-field Gaussian state",
                lambda func: func(
                    {"theta": np.array([0.0, 1.0], dtype=float)},
                    {"theta": np.array([1.0, 0.5], dtype=float)},
                    {"theta": lambda x: x},
                    3,
                    lambda draws: draws,
                    7,
                ),
                _assert_draw_bundle(3, 8),
            ),
            negative=ProbeCase(
                "reject a missing variational mean state",
                lambda func: func(
                    None,
                    {"theta": np.array([1.0, 0.5], dtype=float)},
                    {"theta": lambda x: x},
                    3,
                    lambda draws: draws,
                    7,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.incremental_attention.enable_incremental_state_configuration": ProbePlan(
            positive=ProbeCase(
                "incremental attention decorates a class with state accessors",
                lambda func: func(_DummyBlock),
                _assert_type(type),
            ),
            negative=ProbeCase(
                "incremental attention rejects a missing class",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_subgraph.greedy_maximum_subgraph": ProbePlan(
            positive=ProbeCase(
                "greedy maximum subgraph picks non-conflicting high-score nodes",
                lambda func: func(
                    np.array(
                        [
                            [False, True, False],
                            [True, False, False],
                            [False, False, False],
                        ],
                        dtype=bool,
                    ),
                    np.array([2.0, 1.0, 3.0]),
                ),
                _assert_array(np.array([True, False, True])),
            ),
            negative=ProbeCase(
                "greedy maximum subgraph rejects a missing score vector",
                lambda func: func(np.zeros((2, 2), dtype=bool), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
