"""Kalman filter family runtime probe plans split from the monolithic registry."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    safe_import_module = rt.safe_import_module

    def _kalman_filter_plans() -> dict[str, ProbePlan]:
        def _assert_filter_rs_state(result: Any) -> None:
            assert isinstance(result, dict)
            assert set(result) == {"x", "P"}
            np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([1.0, -1.0], dtype=float))
            np.testing.assert_allclose(np.asarray(result["P"], dtype=float), np.eye(2))

        def _assert_filter_rs_predicted(result: Any) -> None:
            assert isinstance(result, dict)
            np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([1.5, -1.5], dtype=float))
            np.testing.assert_allclose(np.asarray(result["P"], dtype=float), 1.1 * np.eye(2))

        def _assert_filter_rs_steady(result: Any) -> None:
            assert isinstance(result, dict)
            np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([1.5, -1.5], dtype=float))
            np.testing.assert_allclose(np.asarray(result["P"], dtype=float), np.eye(2))

        def _assert_filter_rs_measurement(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            z_pred, innovation = result
            np.testing.assert_allclose(np.asarray(z_pred, dtype=float), np.array([1.0], dtype=float))
            np.testing.assert_allclose(np.asarray(innovation, dtype=float), np.array([0.2], dtype=float))

        def _assert_filter_rs_updated(result: Any) -> None:
            assert isinstance(result, dict)
            np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([1.1, -1.0], dtype=float))
            np.testing.assert_allclose(
                np.asarray(result["P"], dtype=float), np.array([[0.5, 0.0], [0.0, 1.0]], dtype=float)
            )

        def _assert_filter_rs_steady_updated(result: Any) -> None:
            assert isinstance(result, dict)
            np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([1.1, -1.0], dtype=float))
            np.testing.assert_allclose(np.asarray(result["P"], dtype=float), np.eye(2))

        def _assert_static_state(result: Any) -> None:
            assert isinstance(result, dict)
            assert set(result) == {"x", "P", "F", "Q", "H", "R"}
            np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([1.0], dtype=float))
            np.testing.assert_allclose(np.asarray(result["P"], dtype=float), np.array([[1.0]], dtype=float))

        def _assert_static_predicted(result: Any) -> None:
            assert isinstance(result, dict)
            np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([0.5], dtype=float))
            np.testing.assert_allclose(np.asarray(result["P"], dtype=float), np.array([[0.35]], dtype=float))

        def _assert_static_updated(result: Any) -> None:
            assert isinstance(result, dict)
            np.testing.assert_allclose(
                np.asarray(result["x"], dtype=float), np.array([1.09090909], dtype=float), atol=1e-8
            )
            np.testing.assert_allclose(
                np.asarray(result["P"], dtype=float), np.array([[0.09090909]], dtype=float), atol=1e-8
            )

        def _filter_rs_initial_state() -> dict[str, np.ndarray]:
            module = safe_import_module("ageoa.kalman_filters.filter_rs.atoms")
            return module.initializekalmanstatemodel({"x": np.array([1.0, -1.0]), "P": np.eye(2)})

        def _static_initial_state() -> dict[str, np.ndarray]:
            module = safe_import_module("ageoa.kalman_filters.static_kf.atoms")
            return module.initializelineargaussianstatemodel(
                1.0,
                1.0,
                0.5,
                0.1,
                1.0,
                0.1,
            )

        return {
            "ageoa.kalman_filters.filter_rs.initializekalmanstatemodel": ProbePlan(
                positive=ProbeCase(
                    "initialize a deterministic Kalman state bundle",
                    lambda func: func({"x": np.array([1.0, -1.0]), "P": np.eye(2)}),
                    _assert_filter_rs_state,
                ),
                negative=ProbeCase(
                    "reject a missing initialization config",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.filter_rs.predictlatentstateandcovariance": ProbePlan(
                positive=ProbeCase(
                    "predict latent mean and covariance for a linear Gaussian step",
                    lambda func: func(
                        _filter_rs_initial_state(),
                        np.array([0.5, -0.5]),
                        np.eye(2),
                        np.eye(2),
                        0.1 * np.eye(2),
                    ),
                    _assert_filter_rs_predicted,
                ),
                negative=ProbeCase(
                    "reject a missing transition matrix",
                    lambda func: func(
                        _filter_rs_initial_state(),
                        np.array([0.5, -0.5]),
                        np.eye(2),
                        None,
                        0.1 * np.eye(2),
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.filter_rs.predictlatentstatesteadystate": ProbePlan(
                positive=ProbeCase(
                    "predict only the latent mean for a steady-state Kalman step",
                    lambda func: func(_filter_rs_initial_state(), np.array([0.5, -0.5]), np.eye(2)),
                    _assert_filter_rs_steady,
                ),
                negative=ProbeCase(
                    "reject a missing control vector",
                    lambda func: func(_filter_rs_initial_state(), None, np.eye(2)),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.filter_rs.evaluatemeasurementoracle": ProbePlan(
                positive=ProbeCase(
                    "evaluate predicted measurement and innovation",
                    lambda func: func(np.array([1.0, -1.0]), np.array([1.2]), np.array([[1.0, 0.0]])),
                    _assert_filter_rs_measurement,
                ),
                negative=ProbeCase(
                    "reject a missing observation matrix",
                    lambda func: func(np.array([1.0, -1.0]), np.array([1.2]), None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.filter_rs.updateposteriorstateandcovariance": ProbePlan(
                positive=ProbeCase(
                    "update posterior state and covariance from an innovation",
                    lambda func: func(
                        _filter_rs_initial_state(),
                        np.array([1.2]),
                        np.array([[1.0]]),
                        np.array([[1.0, 0.0]]),
                        np.array([0.2]),
                    ),
                    _assert_filter_rs_updated,
                ),
                negative=ProbeCase(
                    "reject a missing innovation vector",
                    lambda func: func(
                        _filter_rs_initial_state(),
                        np.array([1.2]),
                        np.array([[1.0]]),
                        np.array([[1.0, 0.0]]),
                        None,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.filter_rs.updateposteriorstatesteadystate": ProbePlan(
                positive=ProbeCase(
                    "update a steady-state posterior using a precomputed gain",
                    lambda func: func(
                        {"x": np.array([1.0, -1.0]), "P": np.eye(2), "K": np.array([[0.5], [0.0]])},
                        np.array([1.2]),
                        np.array([0.2]),
                    ),
                    _assert_filter_rs_steady_updated,
                ),
                negative=ProbeCase(
                    "reject a missing innovation vector",
                    lambda func: func(
                        {"x": np.array([1.0, -1.0]), "P": np.eye(2), "K": np.array([[0.5], [0.0]])},
                        np.array([1.2]),
                        None,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.static_kf.initializelineargaussianstatemodel": ProbePlan(
                positive=ProbeCase(
                    "initialize a scalar linear Gaussian state-space model",
                    lambda func: func(1.0, 1.0, 0.5, 0.1, 1.0, 0.1),
                    _assert_static_state,
                ),
                negative=ProbeCase(
                    "reject a non-numeric initial state",
                    lambda func: func(np.array([1.0]), 1.0, 0.5, 0.1, 1.0, 0.1),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.static_kf.predictlatentstate": ProbePlan(
                positive=ProbeCase(
                    "predict the latent state for a scalar Kalman model",
                    lambda func: func(_static_initial_state()),
                    _assert_static_predicted,
                ),
                negative=ProbeCase(
                    "reject a missing state model",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.static_kf.updatewithmeasurement": ProbePlan(
                positive=ProbeCase(
                    "update the scalar Kalman model with one measurement",
                    lambda func: func(_static_initial_state(), 1.1),
                    _assert_static_updated,
                ),
                negative=ProbeCase(
                    "reject a missing measurement",
                    lambda func: func(_static_initial_state(), None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.static_kf.exposelatentmean": ProbePlan(
                positive=ProbeCase(
                    "expose the scalar latent mean from the model state",
                    lambda func: func(_static_initial_state()),
                    _assert_array(np.array([1.0], dtype=float)),
                ),
                negative=ProbeCase(
                    "reject a missing state model",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.kalman_filters.static_kf.exposecovariance": ProbePlan(
                positive=ProbeCase(
                    "expose the scalar covariance matrix from the model state",
                    lambda func: func(_static_initial_state()),
                    _assert_array(np.array([[1.0]], dtype=float)),
                ),
                negative=ProbeCase(
                    "reject a missing state model",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
        }

    return _kalman_filter_plans()
