"""Runtime probe plans for advancedvi and institutional quant engine families."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def get_probe_plans() -> dict[str, ProbePlan]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_finite_vector = rt._assert_finite_vector
    _assert_float_mask = rt._assert_float_mask
    _assert_inventory_adjusted_quotes = rt._assert_inventory_adjusted_quotes
    _assert_market_maker_state = rt._assert_market_maker_state
    _assert_nonincreasing_float_list = rt._assert_nonincreasing_float_list
    _assert_profitable_cycles = rt._assert_profitable_cycles
    _assert_scalar = rt._assert_scalar
    _assert_tuple = rt._assert_tuple
    _assert_type = rt._assert_type
    _assert_unit_interval_shape = rt._assert_unit_interval_shape
    safe_import_module = rt.safe_import_module

    def _quadratic_obj(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float)
        return float(np.sum(arr**2))

    def _seeded_hawkes(func: Callable[..., Any]) -> Any:
        state = np.random.get_state()
        try:
            np.random.seed(7)
            return func(0.2, 0.3, 1.5, 2.0)
        finally:
            np.random.set_state(state)

    def _seeded_heston(func: Callable[..., Any]) -> Any:
        state = np.random.get_state()
        try:
            np.random.seed(11)
            return func(100.0, 0.04, 0.1, 1.2, 0.04, 0.3, 1.0, 0.25, 3)
        finally:
            np.random.set_state(state)

    def _assert_hawkes_points(result: Any) -> None:
        arr = np.asarray(result, dtype=float)
        assert arr.ndim == 1
        if arr.size:
            assert np.all(np.diff(arr) >= 0)
            assert np.all(arr > 0.0)
            assert np.all(arr <= 2.0)

    def _assert_heston_paths(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        s_paths, v_paths = result
        s_paths = np.asarray(s_paths, dtype=float)
        v_paths = np.asarray(v_paths, dtype=float)
        assert s_paths.shape == (3, 4)
        assert v_paths.shape == (3, 4)
        assert np.all(s_paths > 0.0)
        assert np.all(v_paths >= 0.0)

    def _assert_advancedvi_gradient_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 4
        grad, value, state_out, rng_out = result
        np.testing.assert_allclose(
            np.asarray(grad, dtype=float),
            np.array([2.0, -4.0], dtype=float),
            atol=1e-4,
        )
        assert np.isclose(float(value), 5.0)
        np.testing.assert_allclose(np.asarray(state_out, dtype=float), np.array([9.0, 8.0], dtype=float))
        assert rng_out == 123

    def _assert_advancedvi_optimization_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        q_out, rng_out, final_state = result
        q_out = np.asarray(q_out, dtype=float)
        final_state = np.asarray(final_state, dtype=float)
        assert q_out.shape == (2,)
        assert final_state.shape == (2,)
        assert np.all(np.isfinite(q_out))
        assert np.all(np.isfinite(final_state))
        np.testing.assert_allclose(q_out, final_state)
        assert rng_out == 123

    def _assert_kalman_init_state(result: Any) -> None:
        assert hasattr(result, "x")
        assert hasattr(result, "p")
        assert hasattr(result, "q")
        assert hasattr(result, "r")
        assert np.isclose(float(result.x), 0.0)
        assert np.isclose(float(result.p), 1.0)
        assert np.isclose(float(result.q), 0.1)
        assert np.isclose(float(result.r), 0.2)

    def _assert_kalman_update_state(result: Any) -> None:
        assert hasattr(result, "x")
        assert hasattr(result, "p")
        assert hasattr(result, "q")
        assert hasattr(result, "r")
        predicted_p = 1.0 + 0.1
        gain = predicted_p / (predicted_p + 0.2)
        assert np.isclose(float(result.x), gain * 2.0)
        assert np.isclose(float(result.p), (1.0 - gain) * predicted_p)
        assert np.isclose(float(result.q), 0.1)
        assert np.isclose(float(result.r), 0.2)

    def _assert_queue_state_init(result: Any) -> None:
        assert hasattr(result, "my_order_id")
        assert hasattr(result, "my_qty")
        assert hasattr(result, "orders_ahead")
        assert hasattr(result, "is_filled")
        assert result.my_order_id == "order-1"
        assert np.isclose(float(result.my_qty), 10.0)
        assert np.isclose(float(result.orders_ahead), 10000.0)
        assert result.is_filled is False

    def _assert_queue_state_update(result: Any) -> None:
        assert hasattr(result, "my_order_id")
        assert hasattr(result, "my_qty")
        assert hasattr(result, "orders_ahead")
        assert hasattr(result, "is_filled")
        assert result.my_order_id == "order-1"
        assert np.isclose(float(result.my_qty), 10.0)
        assert np.isclose(float(result.orders_ahead), 13.0)
        assert result.is_filled is False

    return {
        "ageoa.advancedvi.core.evaluate_log_probability_density": ProbePlan(
            positive=ProbeCase(
                "evaluate a diagonal Gaussian log density on a small vector",
                lambda func: func(np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, -1.0])),
                _assert_scalar(-2.8378770664093453),
            ),
            negative=ProbeCase(
                "reject a missing parameter vector",
                lambda func: func(None, np.array([1.0, -1.0])),
                expect_exception=True,
            ),
        ),
        "ageoa.advancedvi.core.gradient_oracle_evaluation": ProbePlan(
            positive=ProbeCase(
                "estimate a quadratic objective gradient with explicit state threading",
                lambda func: func(
                    123,
                    _quadratic_obj,
                    "fd",
                    np.zeros(2, dtype=float),
                    np.array([9.0, 8.0], dtype=float),
                    np.array([1.0, -2.0], dtype=float),
                    None,
                ),
                _assert_advancedvi_gradient_bundle,
            ),
            negative=ProbeCase(
                "reject a missing objective function",
                lambda func: func(
                    123,
                    None,
                    "fd",
                    np.zeros(2, dtype=float),
                    np.array([9.0, 8.0], dtype=float),
                    np.array([1.0, -2.0], dtype=float),
                    None,
                ),
                expect_exception=True,
                ),
                parity_used=True,
            ),
        "ageoa.advancedvi.core.optimizationlooporchestration": ProbePlan(
            positive=ProbeCase(
                "run a bounded deterministic optimization loop with the scipy fallback path",
                lambda func: func(
                    None,
                    5,
                    _quadratic_obj,
                    np.array([1.0, -2.0], dtype=float),
                    123,
                ),
                _assert_advancedvi_optimization_bundle,
            ),
            negative=ProbeCase(
                "reject a missing objective oracle for the optimization loop",
                lambda func: func(
                    None,
                    5,
                    None,
                    np.array([1.0, -2.0], dtype=float),
                    123,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.hawkes_process.hawkesprocesssimulator": ProbePlan(
            positive=ProbeCase(
                "simulate a Hawkes trajectory with a fixed NumPy seed",
                _seeded_hawkes,
                _assert_hawkes_points,
            ),
            negative=ProbeCase(
                "reject a non-numeric beta value",
                lambda func: func(0.2, 0.3, "bad", 2.0),
                expect_exception=True,
            ),
        ),
        "ageoa.institutional_quant_engine.heston_model.hestonpathsampler": ProbePlan(
            positive=ProbeCase(
                "sample Heston price and variance paths with a fixed NumPy seed",
                _seeded_heston,
                _assert_heston_paths,
            ),
            negative=ProbeCase(
                "reject an invalid correlation coefficient",
                lambda func: func(100.0, 0.04, 2.0, 1.2, 0.04, 0.3, 1.0, 0.25, 3),
                expect_exception=True,
            ),
        ),
        "ageoa.institutional_quant_engine.kalman_filter.kalmanfilterinit": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic scalar Kalman state",
                lambda func: func(0.1, 0.2, 1.0),
                _assert_kalman_init_state,
            ),
            negative=ProbeCase(
                "reject a non-positive process variance",
                lambda func: func(0.0, 0.2, 1.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.kalman_filter.kalmanmeasurementupdate": ProbePlan(
            positive=ProbeCase(
                "apply one scalar Kalman measurement update",
                lambda func: func(
                    safe_import_module("ageoa.institutional_quant_engine.kalman_filter.atoms").kalmanfilterinit(0.1, 0.2, 1.0),
                    2.0,
                ),
                _assert_kalman_update_state,
            ),
            negative=ProbeCase(
                "reject a missing prior state",
                lambda func: func(None, 2.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.queue_estimator.initializeorderstate": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic queue state",
                lambda func: func("order-1", 10.0),
                _assert_queue_state_init,
            ),
            negative=ProbeCase(
                "reject an empty order id",
                lambda func: func("", 10.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.queue_estimator.updatequeueontrade": ProbePlan(
            positive=ProbeCase(
                "advance queue state after consuming queue-ahead volume",
                lambda func: func(
                    safe_import_module("ageoa.institutional_quant_engine.queue_estimator.atoms").initializeorderstate("order-1", 10.0).model_copy(update={"orders_ahead": 25.0}),
                    12.0,
                ),
                _assert_queue_state_update,
            ),
            negative=ProbeCase(
                "reject a negative trade quantity",
                lambda func: func(
                    safe_import_module("ageoa.institutional_quant_engine.queue_estimator.atoms").initializeorderstate("order-1", 10.0).model_copy(update={"orders_ahead": 25.0}),
                    -1.0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.copula_dependence.simulate_copula_dependence": ProbePlan(
            positive=ProbeCase(
                "simulate a deterministic t-copula dependence surface with a fixed NumPy seed",
                lambda func: (
                    np.random.seed(7),
                    func(
                        np.array(
                            [
                                [0.01, -0.02],
                                [0.03, 0.01],
                                [-0.01, 0.02],
                                [0.00, -0.01],
                            ],
                            dtype=float,
                        ),
                        0.25,
                        5,
                    ),
                )[1],
                _assert_unit_interval_shape((4, 2)),
            ),
            negative=ProbeCase(
                "reject a non positive-semidefinite copula correlation",
                lambda func: func(np.array([[0.01, -0.02], [0.03, 0.01]], dtype=float), 1.1, 5),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.dynamic_hedge.kalman_hedge_ratio": ProbePlan(
            positive=ProbeCase(
                "estimate a deterministic rolling hedge ratio for two short price series",
                lambda func: func(
                    np.array([10.0, 10.5, 11.0, 11.8], dtype=float),
                    np.array([9.0, 9.6, 10.1, 10.9], dtype=float),
                    0.05,
                ),
                _assert_array(np.array([0.0, 1.0825628, 1.08819893, 1.08327791], dtype=float), atol=1e-6),
            ),
            negative=ProbeCase(
                "reject a non-numeric state noise ratio",
                lambda func: func(
                    np.array([10.0, 10.5, 11.0], dtype=float),
                    np.array([9.0, 9.6, 10.1], dtype=float),
                    "bad",
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.evt_model.fit_gpd_tail": ProbePlan(
            positive=ProbeCase(
                "fit a deterministic GPD tail on a short return series",
                lambda func: func(
                    np.array([-0.08, -0.06, -0.03, -0.01, 0.01, 0.02, -0.07, -0.02], dtype=float),
                    0.25,
                ),
                _assert_finite_vector(3),
            ),
            negative=ProbeCase(
                "reject a non-numeric threshold quantile",
                lambda func: func(np.array([-0.02, -0.01, 0.01], dtype=float), "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.supply_chain.propagate_supply_shock": ProbePlan(
            positive=ProbeCase(
                "propagate a deterministic supply shock across a small weighted network",
                lambda func: func(
                    np.array(
                        [
                            [0.0, 0.4, 0.0],
                            [0.0, 0.0, 0.5],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=float,
                    ),
                    np.array([1.0, 0.0, 0.0], dtype=float),
                ),
                _assert_array(np.array([1.0, 0.4, 0.2], dtype=float)),
            ),
            negative=ProbeCase(
                "reject incompatible adjacency and shock dimensions",
                lambda func: func(np.eye(3, dtype=float), np.array([1.0, 0.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.triangular_arbitrage.detect_triangular_arbitrage": ProbePlan(
            positive=ProbeCase(
                "detect profitable three-currency arbitrage cycles in a deterministic rate matrix",
                lambda func: func(
                    np.array(
                        [
                            [1.0, 0.9, 1.2],
                            [1.1, 1.0, 1.2],
                            [0.95, 0.8, 1.0],
                        ],
                        dtype=float,
                    )
                ),
                _assert_profitable_cycles(),
            ),
            negative=ProbeCase(
                "reject a one-dimensional rate vector",
                lambda func: func(np.array([1.0, 0.9, 1.2], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.wash_trade.detect_wash_trade_rings": ProbePlan(
            positive=ProbeCase(
                "flag participants in a deterministic directed wash-trade cycle",
                lambda func: func(
                    np.array(
                        [
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        dtype=float,
                    )
                ),
                _assert_float_mask(np.array([1.0, 1.0, 1.0, 0.0], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a one-dimensional trade graph",
                lambda func: func(np.array([0.0, 1.0, 0.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
