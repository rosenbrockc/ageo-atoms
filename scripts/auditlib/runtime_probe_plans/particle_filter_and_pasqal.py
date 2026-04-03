"""Runtime probe plans for particle-filter and Pasqal families."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    safe_import_module = rt.safe_import_module

    def _assert_filter_step_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 5
        prior_state, model_spec, control_t, observation_t, rng_key = result
        assert isinstance(prior_state, dict)
        assert prior_state["rng_seed"] == 7
        assert model_spec == {"transition": "unit"}
        np.testing.assert_allclose(np.asarray(control_t, dtype=float), np.array([0.5], dtype=float))
        np.testing.assert_allclose(np.asarray(observation_t, dtype=float), np.array([1.5], dtype=float))
        np.testing.assert_array_equal(np.asarray(rng_key, dtype=np.int64), np.array([7], dtype=np.int64))

    def _assert_particle_propagation_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        proposed, carry_weights, rng_key_next = result
        expected_noise = np.random.RandomState(5).randn(2)
        np.testing.assert_allclose(
            np.asarray(proposed, dtype=float),
            np.array([1.0, 2.0], dtype=float) + expected_noise,
        )
        np.testing.assert_allclose(
            np.asarray(carry_weights, dtype=float),
            np.array([0.25, 0.75], dtype=float),
        )
        np.testing.assert_array_equal(
            np.asarray(rng_key_next, dtype=np.int64),
            np.array([6], dtype=np.int64),
        )

    def _assert_likelihood_reweight_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        normalized, log_likelihood = result
        particles = np.array([1.0, 2.0], dtype=float)
        carry_weights = np.array([0.4, 0.6], dtype=float)
        obs = 1.5
        log_lik = -0.5 * (particles - obs) ** 2
        log_weights = np.log(carry_weights + 1e-300) + log_lik
        max_lw = np.max(log_weights)
        weights_exp = np.exp(log_weights - max_lw)
        total = weights_exp.sum()
        expected_normalized = weights_exp / total
        expected_log_likelihood = float(max_lw + np.log(total) - np.log(len(particles)))
        np.testing.assert_allclose(np.asarray(normalized, dtype=float), expected_normalized)
        assert abs(float(log_likelihood) - expected_log_likelihood) < 1e-12

    def _assert_particle_filter_result(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        posterior, trace = result
        assert isinstance(posterior, dict)
        assert isinstance(trace, dict)
        assert "particles" in posterior and "weights" in posterior
        assert "log_likelihood" in trace and "ess" in trace

    def _pasqal_positive(func: Callable[..., Any]) -> Any:
        import networkx as nx

        state_mod = safe_import_module("ageoa.pasqal.docking_state")
        state = state_mod.MolecularDockingState()
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        return func(graph, 2, state)

    def _assert_pasqal_result(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        mappings, state = result
        assert isinstance(mappings, list)
        assert len(mappings) == 2
        assert all(isinstance(item, dict) for item in mappings)
        assert hasattr(state, "graph")

    def _pasqal_mwis_positive(func: Callable[..., Any]) -> Any:
        import networkx as nx

        state_mod = safe_import_module("ageoa.pasqal.docking_state")
        state = state_mod.MolecularDockingState()
        graph = nx.Graph()
        graph.add_nodes_from(
            [
                (0, {"weight": 1.0}),
                (1, {"weight": 2.0}),
                (2, {"weight": 1.5}),
            ]
        )
        graph.add_edges_from([(0, 1), (1, 2)])
        lattice = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}
        return func(graph, lattice, 2, state)

    def _assert_pasqal_mwis_result(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        solutions, state = result
        assert isinstance(solutions, list)
        assert len(solutions) == 2
        assert all(solution == {0, 2} for solution in solutions)
        assert getattr(state, "graph", None) is not None
        assert getattr(state, "lattice_id_coord_dic", None) == {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}

    return {
        "ageoa.particle_filters.basic.filter_step_preparation_and_dispatch": ProbePlan(
            positive=ProbeCase(
                "prepare a deterministic particle-filter step bundle from prior state",
                lambda func: func(
                    {
                        "particles": np.array([1.0, 2.0], dtype=float),
                        "weights": np.array([0.4, 0.6], dtype=float),
                        "rng_seed": 7,
                    },
                    {"transition": "unit"},
                    np.array([0.5], dtype=float),
                    np.array([1.5], dtype=float),
                ),
                _assert_filter_step_bundle,
            ),
            negative=ProbeCase(
                "reject a missing prior state bundle",
                lambda func: func(None, {"transition": "unit"}, np.array([0.5], dtype=float), np.array([1.5], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.particle_filters.basic.particle_propagation_kernel": ProbePlan(
            positive=ProbeCase(
                "propagate a deterministic particle pair with fixed RNG state",
                lambda func: func(
                    {
                        "particles": np.array([1.0, 2.0], dtype=float),
                        "weights": np.array([0.25, 0.75], dtype=float),
                    },
                    {"transition": "unit"},
                    np.array([0.5], dtype=float),
                    np.array([5], dtype=np.int64),
                ),
                _assert_particle_propagation_bundle,
            ),
            negative=ProbeCase(
                "reject a missing prior state during propagation",
                lambda func: func(None, {"transition": "unit"}, np.array([0.5], dtype=float), np.array([5], dtype=np.int64)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.particle_filters.basic.likelihood_reweight_kernel": ProbePlan(
            positive=ProbeCase(
                "reweight a deterministic two-particle proposal against a scalar observation",
                lambda func: func(
                    np.array([1.0, 2.0], dtype=float),
                    np.array([0.4, 0.6], dtype=float),
                    np.array([1.5], dtype=float),
                    {"likelihood": "gaussian"},
                ),
                _assert_likelihood_reweight_bundle,
            ),
            negative=ProbeCase(
                "reject a missing proposed particle array",
                lambda func: func(None, np.array([0.4, 0.6], dtype=float), np.array([1.5], dtype=float), {"likelihood": "gaussian"}),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.particle_filters.basic.resample_and_belief_projection": ProbePlan(
            positive=ProbeCase(
                "resample weighted particles into a posterior belief state",
                lambda func: func(
                    np.array([10.0, 20.0, 30.0]),
                    np.array([0.2, 0.3, 0.5]),
                    np.array([4], dtype=np.int64),
                    -1.25,
                ),
                _assert_particle_filter_result,
            ),
            negative=ProbeCase(
                "reject a non-numeric log likelihood",
                lambda func: func(
                    np.array([10.0, 20.0, 30.0]),
                    np.array([0.2, 0.3, 0.5]),
                    np.array([4], dtype=np.int64),
                    "bad",
                ),
                expect_exception=True,
            ),
        ),
        "ageoa.pasqal.docking.sub_graph_embedder": ProbePlan(
            positive=ProbeCase(
                "extract deterministic subgraph mappings from a small graph",
                _pasqal_positive,
                _assert_pasqal_result,
            ),
            negative=ProbeCase(
                "reject a non-positive subgraph quantity",
                lambda func: func(
                    __import__("networkx").Graph(),
                    0,
                    safe_import_module("ageoa.pasqal.docking_state").MolecularDockingState(),
                ),
                expect_exception=True,
            ),
        ),
        "ageoa.pasqal.docking.quantum_mwis_solver": ProbePlan(
            positive=ProbeCase(
                "solve a deterministic MWIS instance over a small weighted graph",
                _pasqal_mwis_positive,
                _assert_pasqal_mwis_result,
            ),
            negative=ProbeCase(
                "reject a non-positive sample count",
                lambda func: func(
                    __import__("networkx").Graph(),
                    {},
                    0,
                    safe_import_module("ageoa.pasqal.docking_state").MolecularDockingState(),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
