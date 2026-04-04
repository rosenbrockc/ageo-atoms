"""Pronto family runtime probe plans split from the monolithic registry."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pronto_blip_filter_plans(ProbeCase: type, ProbePlan: type, helpers: dict[str, Any]) -> dict[str, Any]:
    signal = np.zeros(2000, dtype=float)
    peak_positions = np.array([300, 700, 1100, 1500, 1900], dtype=int)
    signal[peak_positions] = np.array([1.0, 1.2, 0.9, 1.1, 1.0], dtype=float)
    rpeaks = np.array([300, 700, 1100, 1500], dtype=int)

    return {
        "ageoa.pronto.blip_filter.bandpass_filter": ProbePlan(
            positive=ProbeCase(
                "Pronto blip-filter bandpass preserves waveform length on a synthetic ECG-like trace",
                lambda func: func(signal),
                helpers["_assert_shape"](signal.shape),
            ),
            negative=ProbeCase(
                "Pronto blip-filter bandpass rejects a missing signal",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.blip_filter.r_peak_detection": ProbePlan(
            positive=ProbeCase(
                "Pronto blip-filter peak detection returns monotonic peak indices",
                lambda func: func(signal),
                helpers["_assert_monotonic_index_array"](max_value=len(signal) - 1),
            ),
            negative=ProbeCase(
                "Pronto blip-filter peak detection rejects a missing filtered signal",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.blip_filter.peak_correction": ProbePlan(
            positive=ProbeCase(
                "Pronto blip-filter peak correction returns monotonic corrected peaks",
                lambda func: func(signal, rpeaks),
                helpers["_assert_monotonic_index_array"](max_value=len(signal) - 1),
            ),
            negative=ProbeCase(
                "Pronto blip-filter peak correction rejects a missing filtered signal",
                lambda func: func(None, rpeaks),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.blip_filter.template_extraction": ProbePlan(
            positive=ProbeCase(
                "Pronto blip-filter template extraction returns templates and aligned peaks",
                lambda func: func(signal, rpeaks),
                helpers["_assert_pair_of_arrays"](),
            ),
            negative=ProbeCase(
                "Pronto blip-filter template extraction rejects a missing filtered signal",
                lambda func: func(None, rpeaks),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.blip_filter.heart_rate_computation": ProbePlan(
            positive=ProbeCase(
                "Pronto blip-filter heart-rate computation returns aligned index and bpm arrays",
                lambda func: func(rpeaks),
                helpers["_assert_pair_of_arrays"](),
            ),
            negative=ProbeCase(
                "Pronto blip-filter heart-rate computation rejects a missing peak series",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _pronto_state_readout_plans(ProbeCase: type, ProbePlan: type, helpers: dict[str, Any]) -> dict[str, Any]:
    yaw_state = {
        "is_robot_standing": True,
        "joint_angles_init": np.array([0.1, -0.2, 0.3], dtype=float),
    }
    leg_state = {
        "xd_b": np.array([0.25, -0.5, 0.75], dtype=float),
        "vel_cov": np.diag([1.0, 2.0, 3.0]).astype(float),
    }

    def _assert_none(result: Any) -> None:
        assert result is None

    def _assert_mode_snapshot(result: Any) -> None:
        assert result == ("stance", "swing")

    def _assert_velocity_readout(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        velocity, velocity_covariance = result
        helpers["_assert_array"](np.array([0.25, -0.5, 0.75], dtype=float))(velocity)
        helpers["_assert_array"](np.diag([1.0, 2.0, 3.0]).astype(float))(velocity_covariance)

    def _assert_pose_snapshot(result: Any) -> None:
        assert isinstance(result, dict)
        assert set(result) == {"position", "orientation"}
        helpers["_assert_array"](np.zeros(3, dtype=float))(result["position"])
        helpers["_assert_array"](np.eye(3, dtype=float))(result["orientation"])

    def _assert_ekf_state(result: Any) -> None:
        assert isinstance(result, dict)
        assert set(result) == {"x", "P", "A", "Q", "H", "R", "x_history", "P_history", "x_pred_history", "P_pred_history"}
        helpers["_assert_array"](np.zeros(6, dtype=float))(result["x"])
        helpers["_assert_array"](np.eye(6, dtype=float) * 1e2)(result["P"])
        helpers["_assert_array"](np.eye(6, dtype=float) * 1e-3)(result["Q"])
        helpers["_assert_array"](np.eye(6, dtype=float))(result["H"])
        helpers["_assert_array"](np.eye(6, dtype=float) * 1e-1)(result["R"])
        assert result["x_history"] == []
        assert result["P_history"] == []
        assert result["x_pred_history"] == []
        assert result["P_pred_history"] == []

    def _assert_yaw_lock_init(result: Any) -> None:
        assert isinstance(result, dict)
        assert set(result) == {
            "correction_period",
            "yaw_slip_detect",
            "yaw_slip_threshold_degrees",
            "yaw_slip_disable_period",
            "is_robot_standing",
            "joint_name",
            "joint_position",
            "joint_angles_init",
            "left_standing_link",
            "right_standing_link",
        }
        assert result["correction_period"] == 1.0
        assert result["yaw_slip_detect"] is False
        assert result["yaw_slip_threshold_degrees"] == 5.0
        assert result["yaw_slip_disable_period"] == 1.0
        assert result["is_robot_standing"] is False
        assert result["joint_name"] is None
        assert result["joint_position"] is None
        assert result["joint_angles_init"] is None
        assert result["left_standing_link"] is None
        assert result["right_standing_link"] is None

    return {
        "ageoa.pronto.ekf_smoother.stateestimatorinit": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic EKF smoother state bundle",
                lambda func: func(),
                _assert_ekf_state,
            ),
            negative=ProbeCase(
                "reject unexpected positional arguments for the zero-argument EKF smoother initializer",
                lambda func: func(1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.foot_contact.foot_sensing_state_update": ProbePlan(
            positive=ProbeCase(
                "merge a foot sensing command into an immutable state snapshot",
                lambda func: func({"left": False, "right": True}, {"left": True}),
                helpers["_assert_value"]({"left": True, "right": True}),
            ),
            negative=ProbeCase(
                "reject a missing sensing command",
                lambda func: func({"left": False}, None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.foot_contact.mode_snapshot_readout": ProbePlan(
            positive=ProbeCase(
                "read current and previous gait modes from immutable foot-contact state",
                lambda func: func({"mode": "stance", "previous_mode": "swing"}),
                _assert_mode_snapshot,
            ),
            negative=ProbeCase(
                "reject a missing mode-state snapshot",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.inverse_schmitt.inverse_schmitt_trigger_transform": ProbePlan(
            positive=ProbeCase(
                "apply inverse Schmitt trigger hysteresis to a simple analog trace",
                lambda func: func(np.array([0.2, 0.4, 0.8, 0.2], dtype=float)),
                helpers["_assert_array"](np.array([1.0, 1.0, 0.0, 1.0], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing input signal",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.leg_odometer.velocitystatereadout": ProbePlan(
            positive=ProbeCase(
                "read body velocity and covariance from immutable leg-odometer state",
                lambda func: func(leg_state),
                _assert_velocity_readout,
            ),
            negative=ProbeCase(
                "reject a missing leg-odometer state snapshot",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.leg_odometer.posequeryaccessors": ProbePlan(
            positive=ProbeCase(
                "return deterministic pose query accessors with identity defaults",
                lambda func: func(),
                _assert_pose_snapshot,
            ),
            negative=ProbeCase(
                "reject unexpected positional arguments for the stateless pose query accessor",
                lambda func: func(1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.torque_adjustment.torqueadjustmentidentitystage": ProbePlan(
            positive=ProbeCase(
                "identity torque-adjustment stage returns no output",
                lambda func: func(),
                _assert_none,
            ),
            negative=ProbeCase(
                "identity torque-adjustment stage rejects unexpected positional arguments",
                lambda func: func(1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.yaw_lock.readrobotstandingstatus": ProbePlan(
            positive=ProbeCase(
                "read the robot-standing status flag from immutable yaw-lock state",
                lambda func: func(yaw_state),
                helpers["_assert_value"](True),
            ),
            negative=ProbeCase(
                "reject a missing yaw-lock state",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.yaw_lock.initializeyawlockstate": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic yaw-lock state bundle",
                lambda func: func(),
                _assert_yaw_lock_init,
            ),
            negative=ProbeCase(
                "reject unexpected positional arguments for the zero-argument yaw-lock initializer",
                lambda func: func(1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.yaw_lock.readinitialjointangles": ProbePlan(
            positive=ProbeCase(
                "read the stored initial joint-angle vector from immutable yaw-lock state",
                lambda func: func(yaw_state),
                helpers["_assert_array"](np.array([0.1, -0.2, 0.3], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing yaw-lock state",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.state_estimator.update_state_estimate": ProbePlan(
            positive=ProbeCase(
                "apply one deterministic EKF-style state update with identity observation model",
                lambda func: func(
                    np.array([0.0, 0.0], dtype=float),
                    np.eye(2, dtype=float),
                    np.array([1.0, -1.0], dtype=float),
                    123456,
                ),
                helpers["_assert_array"](
                    np.array([0.9900990099009901, -0.9900990099009901], dtype=float)
                ),
            ),
            negative=ProbeCase(
                "reject a missing prior state",
                lambda func: func(
                    None,
                    np.eye(2, dtype=float),
                    np.array([1.0, -1.0], dtype=float),
                    123456,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _pronto_dynamic_stance_d12_plans(ProbeCase: type, ProbePlan: type, helpers: dict[str, Any]) -> dict[str, Any]:
    safe_import_module = helpers["safe_import_module"]

    def _assert_stance_state(result: Any) -> None:
        assert isinstance(result, dict)
        assert set(result) == {"stance", "force_threshold", "n_legs", "grf_history"}
        np.testing.assert_allclose(np.asarray(result["stance"], dtype=float), np.zeros(4, dtype=float))
        np.testing.assert_allclose(np.asarray(result["grf_history"], dtype=float), np.zeros(4, dtype=float))

    def _assert_stance_update(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        state_out, stance = result
        assert isinstance(state_out, dict)
        stance_arr = np.asarray(stance, dtype=float)
        np.testing.assert_allclose(stance_arr, np.array([0.0, 1.0, 1.0, 0.0], dtype=float))
        np.testing.assert_allclose(np.asarray(state_out["stance"], dtype=float), stance_arr)
        np.testing.assert_allclose(
            np.asarray(state_out["grf_history"], dtype=float),
            np.array([30.0, 55.0, 80.0, 10.0], dtype=float),
        )

    def _stance_state() -> dict[str, np.ndarray]:
        module = safe_import_module("ageoa.pronto.dynamic_stance_estimator_d12.atoms")
        return module.stancestateinit({"n_legs": 4, "force_threshold": 50.0})

    return {
        "ageoa.pronto.dynamic_stance_estimator_d12.stancestateinit": ProbePlan(
            positive=ProbeCase(
                "initialize deterministic dynamic stance estimator state",
                lambda func: func({"n_legs": 4, "force_threshold": 50.0}),
                _assert_stance_state,
            ),
            negative=ProbeCase(
                "reject a missing config mapping",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.dynamic_stance_estimator_d12.stanceestimation": ProbePlan(
            positive=ProbeCase(
                "estimate stance from a thresholded ground-reaction-force observation",
                lambda func: func(
                    _stance_state(),
                    np.array([30.0, 55.0, 80.0, 10.0], dtype=float),
                ),
                _assert_stance_update,
            ),
            negative=ProbeCase(
                "reject a missing observation vector",
                lambda func: func(_stance_state(), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _pronto_dynamic_stance_plans(ProbeCase: type, ProbePlan: type, helpers: dict[str, Any]) -> dict[str, Any]:
    _assert_value = helpers["_assert_value"]

    def _assert_filter_init(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        state, params = result
        assert set(state) == {"x", "P"}
        assert set(params) == {"A", "H", "Q", "R"}
        np.testing.assert_allclose(np.asarray(state["x"], dtype=float), np.array([[0.0], [1.0]], dtype=float))
        np.testing.assert_allclose(np.asarray(state["P"], dtype=float), np.eye(2, dtype=float))

    def _filter_fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        x0 = np.array([[0.0], [1.0]], dtype=float)
        p0 = np.eye(2, dtype=float)
        A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        Q = 0.1 * np.eye(2, dtype=float)
        R = np.array([[0.5]], dtype=float)
        state = {"x": x0, "P": p0}
        params = {"A": A, "H": H, "Q": Q, "R": R}
        return state, params

    def _assert_predict(result: Any) -> None:
        assert isinstance(result, dict)
        np.testing.assert_allclose(np.asarray(result["x"], dtype=float), np.array([[1.0], [1.0]], dtype=float))
        np.testing.assert_allclose(
            np.asarray(result["P"], dtype=float),
            np.array([[2.1, 1.0], [1.0, 1.1]], dtype=float),
        )

    def _assert_update(result: Any) -> None:
        assert isinstance(result, dict)
        np.testing.assert_allclose(
            np.asarray(result["x"], dtype=float),
            np.array([[1.1615384615384614], [1.0769230769230769]], dtype=float),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(result["P"], dtype=float),
            np.array([[0.40384615384615374, 0.1923076923076923], [0.19230769230769226, 0.7153846153846153]], dtype=float),
            atol=1e-9,
        )

    return {
        "ageoa.pronto.dynamic_stance_estimator.initializefilter": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic Kalman-style stance estimator state and parameter bundle",
                lambda func: func(
                    np.array([[0.0], [1.0]], dtype=float),
                    np.eye(2, dtype=float),
                    np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float),
                    np.array([[1.0, 0.0]], dtype=float),
                    0.1 * np.eye(2, dtype=float),
                    np.array([[0.5]], dtype=float),
                ),
                _assert_filter_init,
            ),
            negative=ProbeCase(
                "reject a missing initial state vector",
                lambda func: func(
                    None,
                    np.eye(2, dtype=float),
                    np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float),
                    np.array([[1.0, 0.0]], dtype=float),
                    0.1 * np.eye(2, dtype=float),
                    np.array([[0.5]], dtype=float),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.dynamic_stance_estimator.predictstep": ProbePlan(
            positive=ProbeCase(
                "propagate the deterministic stance state one step forward",
                lambda func: func(*_filter_fixture(), 0.1),
                _assert_predict,
            ),
            negative=ProbeCase(
                "reject a non-numeric timestep",
                lambda func: func(*_filter_fixture(), "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.dynamic_stance_estimator.updatestep": ProbePlan(
            positive=ProbeCase(
                "update the deterministic stance state with one scalar measurement",
                lambda func: func(
                    {
                        "x": np.array([[1.0], [1.0]], dtype=float),
                        "P": np.array([[2.1, 1.0], [1.0, 1.1]], dtype=float),
                    },
                    _filter_fixture()[1],
                    np.array([[1.2]], dtype=float),
                ),
                _assert_update,
            ),
            negative=ProbeCase(
                "reject a missing measurement",
                lambda func: func(
                    {
                        "x": np.array([[1.0], [1.0]], dtype=float),
                        "P": np.array([[2.1, 1.0], [1.0, 1.1]], dtype=float),
                    },
                    _filter_fixture()[1],
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.dynamic_stance_estimator.querystance": ProbePlan(
            positive=ProbeCase(
                "read the stance scalar from the first state component",
                lambda func: func({"x": np.array([[1.1615384615384614], [1.0769230769230769]], dtype=float), "P": np.eye(2, dtype=float)}),
                _assert_value(1.1615384615384614),
            ),
            negative=ProbeCase(
                "reject a missing current state",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def get_probe_plans() -> dict[str, Any]:
    from ..runtime_probes import (
        ProbeCase,
        ProbePlan,
        _assert_array,
        _assert_monotonic_index_array,
        _assert_pair_of_arrays,
        _assert_shape,
        _assert_value,
        safe_import_module,
    )

    helpers = {
        "_assert_array": _assert_array,
        "_assert_monotonic_index_array": _assert_monotonic_index_array,
        "_assert_pair_of_arrays": _assert_pair_of_arrays,
        "_assert_shape": _assert_shape,
        "_assert_value": _assert_value,
        "safe_import_module": safe_import_module,
    }

    plans: dict[str, Any] = {}
    plans.update(_pronto_blip_filter_plans(ProbeCase, ProbePlan, helpers))
    plans.update(_pronto_dynamic_stance_plans(ProbeCase, ProbePlan, helpers))
    plans.update(_pronto_state_readout_plans(ProbeCase, ProbePlan, helpers))
    plans.update(_pronto_dynamic_stance_d12_plans(ProbeCase, ProbePlan, helpers))
    return plans
