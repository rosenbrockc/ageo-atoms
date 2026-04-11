"""Runtime probe plans for rust_robotics families."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    safe_import_module = rt.safe_import_module

    def _assert_vector(expected: np.ndarray):
        def _assert(result: Any) -> None:
            np.testing.assert_allclose(np.asarray(result, dtype=float), expected)

        return _assert

    def _assert_tuple_of_arrays(expected_first: np.ndarray, expected_second: np.ndarray):
        def _assert(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            first, second = result
            np.testing.assert_allclose(np.asarray(first, dtype=float), expected_first)
            np.testing.assert_allclose(np.asarray(second, dtype=float), expected_second)

        return _assert

    def _assert_bicycle_dynamics_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        x_dot, jacobian, u_inferred = result
        np.testing.assert_allclose(np.asarray(x_dot, dtype=float), np.array([2.0, 0.0, 0.0, 0.5]))
        assert np.asarray(jacobian, dtype=float).shape == (4, 4)
        np.testing.assert_allclose(np.asarray(u_inferred, dtype=float), np.array([0.0, 0.5]))

    def _assert_geometry_spec(result: Any) -> None:
        assert result == {"lf": 1.2, "lr": 1.3, "L": 2.5}

    def _assert_geometry_tuple(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        np.testing.assert_allclose(np.asarray(result, dtype=float), np.array([1.2, 1.3, 2.5], dtype=float))

    def _assert_sideslip_angle(result: Any) -> None:
        expected = float(np.arctan(1.3 / 2.5 * np.tan(0.2)))
        np.testing.assert_allclose(float(result), expected)

    def _assert_linearized_matrices(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        A, B = result
        model = {"lf": 1.2, "lr": 1.3, "L": 2.5}
        x = np.array([0.0, 0.0, 0.0, 2.0], dtype=float)
        u = np.array([0.1, 0.5], dtype=float)
        theta = x[2]
        v = x[3]
        delta = u[0]
        beta = np.arctan(model["lr"] / model["L"] * np.tan(delta))
        dbeta_ddelta = (model["lr"] / model["L"]) / (np.cos(delta) ** 2) / (
            1.0 + (model["lr"] / model["L"] * np.tan(delta)) ** 2
        )
        expected_A = np.zeros((4, 4))
        expected_A[0, 2] = -v * np.sin(theta + beta)
        expected_A[0, 3] = np.cos(theta + beta)
        expected_A[1, 2] = v * np.cos(theta + beta)
        expected_A[1, 3] = np.sin(theta + beta)
        expected_A[2, 3] = np.sin(beta) / model["lr"]
        expected_B = np.zeros((4, 2))
        expected_B[0, 0] = -v * np.sin(theta + beta) * dbeta_ddelta
        expected_B[1, 0] = v * np.cos(theta + beta) * dbeta_ddelta
        expected_B[2, 0] = v * np.cos(beta) * dbeta_ddelta / model["lr"]
        expected_B[3, 1] = 1.0
        np.testing.assert_allclose(np.asarray(A, dtype=float), expected_A)
        np.testing.assert_allclose(np.asarray(B, dtype=float), expected_B)

    def _assert_initialize_model(result: Any) -> None:
        assert result == {
            "mass": 1500.0,
            "area_frontal": 2.2,
            "Cd": 0.3,
            "rho": 1.225,
            "Cr": 0.01,
            "g": 9.81,
        }

    def _assert_aero_force(result: Any) -> None:
        np.testing.assert_allclose(float(result), 1.617, atol=1e-12)

    def _assert_rolling_force(result: Any) -> None:
        np.testing.assert_allclose(float(result), 147.15, atol=1e-12)

    def _assert_gravity_force(result: Any) -> None:
        np.testing.assert_allclose(float(result), 0.0, atol=1e-12)

    def _assert_dynamics_derivatives(result: Any) -> None:
        expected = np.array([10.0, (5.0 - 40.425 - 147.15) / 1500.0], dtype=float)
        np.testing.assert_allclose(np.asarray(result, dtype=float), expected)

    def _assert_linearize_dynamics(result: Any) -> None:
        expected = np.array([[0.0, 1.0], [0.0, -0.00539]], dtype=float)
        np.testing.assert_allclose(np.asarray(result, dtype=float), expected, atol=1e-5)

    def _assert_control_synthesis(result: Any) -> None:
        np.testing.assert_allclose(np.asarray(result, dtype=float), np.array([-10.0, 10.0], dtype=float))

    def _assert_inverse_control(result: Any) -> None:
        np.testing.assert_allclose(np.asarray(result, dtype=float), np.array([187.575, 0.0], dtype=float), atol=1e-12)

    def _assert_deserialized_vehicle_model(result: Any) -> None:
        assert result == {
            "mass": 1600.0,
            "area_frontal": 2.3,
            "Cd": 0.3,
            "rho": 1.225,
            "Cr": 0.01,
            "g": 9.81,
        }

    def _assert_modelspec_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        link_lengths, total_reach = result
        np.testing.assert_allclose(np.asarray(link_lengths, dtype=float), np.array([1.0, 2.0, 3.0], dtype=float))
        np.testing.assert_allclose(float(total_reach), 6.0)

    def _assert_ik_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        ik_state, pos_err, goal_dist_sq = result
        np.testing.assert_allclose(np.asarray(ik_state, dtype=float), np.array([0.2, -0.1], dtype=float))
        np.testing.assert_allclose(np.asarray(pos_err, dtype=float), np.array([0.5, 0.5], dtype=float))
        np.testing.assert_allclose(float(goal_dist_sq), 1.0, atol=1e-12)

    def _assert_joint_kernel(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        x_dot, jacobian = result
        np.testing.assert_allclose(np.asarray(x_dot, dtype=float), np.array([1.0, 2.0], dtype=float))
        np.testing.assert_allclose(np.asarray(jacobian, dtype=float), np.eye(2))

    def _positive_temp_json(payload: dict[str, Any], func: Any) -> Any:
        with tempfile.TemporaryDirectory(prefix="rust_robotics_") as tmpdir:
            path = Path(tmpdir) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return func(str(path))

    def _prepare_longitudinal_model() -> None:
        module = safe_import_module("ageoa.rust_robotics.longitudinal_dynamics.atoms")
        module.initialize_model(1500.0, 2.2)

    return {
        "ageoa.rust_robotics.atoms.n_joint_arm_solver": ProbePlan(
            positive=ProbeCase(
                "solve a deterministic planar arm summary from a zero-angle seed",
                lambda func: func(np.array([0.0, 0.0], dtype=float)),
                _assert_vector(np.array([2.0, 0.0], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing joint-angle tensor",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.atoms.dijkstra_path_planning": ProbePlan(
            positive=ProbeCase(
                "compute a deterministic shortest-path distance profile",
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
                _assert_vector(np.array([0.0, 1.0, 3.0], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing adjacency matrix",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.bicycle_kinematic.constructgeometrymodel": ProbePlan(
            positive=ProbeCase(
                "construct a deterministic bicycle geometry model",
                lambda func: func(1.2, 1.3),
                _assert_geometry_spec,
            ),
            negative=ProbeCase(
                "reject a missing axle length",
                lambda func: func(None, 1.3),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.bicycle_kinematic.loadmodelfromfile": ProbePlan(
            positive=ProbeCase(
                "load bicycle geometry from a JSON model file",
                lambda func: _positive_temp_json({"lf": 1.2, "lr": 1.3, "L": 2.5}, func),
                _assert_geometry_spec,
            ),
            negative=ProbeCase(
                "reject a missing filename",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.bicycle_kinematic.querygeometryparameters": ProbePlan(
            positive=ProbeCase(
                "query bicycle geometry parameters from a constructed model",
                lambda func: func({"lf": 1.2, "lr": 1.3, "L": 2.5}),
                _assert_geometry_tuple,
            ),
            negative=ProbeCase(
                "reject a missing geometry model",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.bicycle_kinematic.computesideslipangle": ProbePlan(
            positive=ProbeCase(
                "compute a deterministic bicycle sideslip angle",
                lambda func: func({"lf": 1.2, "lr": 1.3, "L": 2.5}, 0.2),
                _assert_sideslip_angle,
            ),
            negative=ProbeCase(
                "reject a non-numeric steering angle",
                lambda func: func({"lf": 1.2, "lr": 1.3, "L": 2.5}, "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.bicycle_kinematic.computelinearizedstatematrices": ProbePlan(
            positive=ProbeCase(
                "compute the local bicycle linearization matrices",
                lambda func: func(
                    {"lf": 1.2, "lr": 1.3, "L": 2.5},
                    np.array([0.0, 0.0, 0.0, 2.0], dtype=float),
                    np.array([0.1, 0.5], dtype=float),
                ),
                _assert_linearized_matrices,
            ),
            negative=ProbeCase(
                "reject a missing control vector",
                lambda func: func(
                    {"lf": 1.2, "lr": 1.3, "L": 2.5},
                    np.array([0.0, 0.0, 0.0, 2.0], dtype=float),
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.bicycle_kinematic.evaluateandinvertdynamics": ProbePlan(
            positive=ProbeCase(
                "evaluate bicycle kinematics and recover the acceleration control component",
                lambda func: func(
                    {"lf": 1.2, "lr": 1.3, "L": 2.5},
                    np.array([0.0, 0.0, 0.0, 2.0], dtype=float),
                    np.array([0.0, 0.5], dtype=float),
                    0.0,
                    np.array([2.0, 0.0, 0.0, 0.5], dtype=float),
                ),
                _assert_bicycle_dynamics_bundle,
            ),
            negative=ProbeCase(
                "reject a non-numeric evaluation time",
                lambda func: func(
                    {"lf": 1.2, "lr": 1.3, "L": 2.5},
                    np.array([0.0, 0.0, 0.0, 2.0], dtype=float),
                    np.array([0.0, 0.5], dtype=float),
                    "bad",
                    np.array([2.0, 0.0, 0.0, 0.5], dtype=float),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.initialize_model": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic longitudinal-dynamics vehicle model",
                lambda func: func(1500.0, 2.2),
                _assert_initialize_model,
            ),
            negative=ProbeCase(
                "reject a missing mass parameter",
                lambda func: func(None, 2.2),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.compute_aerodynamic_force": ProbePlan(
            positive=ProbeCase(
                "compute aerodynamic drag for a fixed velocity",
                lambda func: (_prepare_longitudinal_model(), func(2.0))[1],
                _assert_aero_force,
            ),
            negative=ProbeCase(
                "reject a missing velocity",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.compute_rolling_force": ProbePlan(
            positive=ProbeCase(
                "compute rolling resistance on a level grade",
                lambda func: (_prepare_longitudinal_model(), func(0.0))[1],
                _assert_rolling_force,
            ),
            negative=ProbeCase(
                "reject a missing grade angle",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.compute_gravity_grade_force": ProbePlan(
            positive=ProbeCase(
                "compute grade gravity force on a flat road",
                lambda func: (_prepare_longitudinal_model(), func(0.0))[1],
                _assert_gravity_force,
            ),
            negative=ProbeCase(
                "reject a missing grade angle",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.evaluate_dynamics_derivatives": ProbePlan(
            positive=ProbeCase(
                "evaluate longitudinal dynamics derivatives deterministically",
                lambda func: (
                    _prepare_longitudinal_model(),
                    func(np.array([0.0, 10.0], dtype=float), np.array([5.0, 0.0], dtype=float), 0.0),
                )[1],
                _assert_dynamics_derivatives,
            ),
            negative=ProbeCase(
                "reject a missing state vector",
                lambda func: func(None, np.array([5.0, 0.0], dtype=float), 0.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.linearize_dynamics": ProbePlan(
            positive=ProbeCase(
                "linearize longitudinal dynamics around a simple operating point",
                lambda func: (
                    _prepare_longitudinal_model(),
                    func(np.array([0.0, 10.0], dtype=float), np.array([5.0, 0.0], dtype=float), 0.0),
                )[1],
                _assert_linearize_dynamics,
            ),
            negative=ProbeCase(
                "reject a missing state vector",
                lambda func: func(None, np.array([5.0, 0.0], dtype=float), 0.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.solve_control_for_target_derivative": ProbePlan(
            positive=ProbeCase(
                "solve a deterministic inverse control target",
                lambda func: (
                    _prepare_longitudinal_model(),
                    func(np.array([0.0, 10.0], dtype=float), np.array([10.0, 0.0], dtype=float), 0.0),
                )[1],
                _assert_inverse_control,
            ),
            negative=ProbeCase(
                "reject a missing desired derivative vector",
                lambda func: func(np.array([0.0, 10.0], dtype=float), None, 0.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.longitudinal_dynamics.deserialize_model_spec": ProbePlan(
            positive=ProbeCase(
                "deserialize a deterministic longitudinal vehicle model file",
                lambda func: _positive_temp_json({"mass": 1600.0, "area_frontal": 2.3}, func),
                _assert_deserialized_vehicle_model,
            ),
            negative=ProbeCase(
                "reject a missing filename",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.n_joint_arm_2d.modelspecloadingandsizing": ProbePlan(
            positive=ProbeCase(
                "load a deterministic n-joint arm model spec",
                lambda func: _positive_temp_json({"link_lengths": [1.0, 2.0, 3.0]}, func),
                _assert_modelspec_bundle,
            ),
            negative=ProbeCase(
                "reject a missing filename",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.n_joint_arm_2d.kinematicgoalfeasibility": ProbePlan(
            positive=ProbeCase(
                "compute inverse-kinematic feasibility and goal distance deterministically",
                lambda func: func(
                    np.array([0.2, -0.1], dtype=float),
                    np.array([1.5, 2.5], dtype=float),
                    np.array([0.0, 0.0], dtype=float),
                    np.array([1.0, 2.0], dtype=float),
                    np.array([1.0, 3.0], dtype=float),
                ),
                _assert_ik_bundle,
            ),
            negative=ProbeCase(
                "reject a missing desired-angle vector",
                lambda func: func(
                    None,
                    np.array([1.5, 2.5], dtype=float),
                    np.array([0.0, 0.0], dtype=float),
                    np.array([1.0, 2.0], dtype=float),
                    np.array([1.0, 3.0], dtype=float),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.n_joint_arm_2d.dynamicsandlinearizationkernel": ProbePlan(
            positive=ProbeCase(
                "compute a deterministic joint-dynamics kernel",
                lambda func: func(
                    np.array([0.0, 0.0], dtype=float),
                    np.array([1.0, 2.0], dtype=float),
                    0.0,
                ),
                _assert_joint_kernel,
            ),
            negative=ProbeCase(
                "reject a missing control vector",
                lambda func: func(
                    np.array([0.0, 0.0], dtype=float),
                    None,
                    0.0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.rust_robotics.n_joint_arm_2d.controlinputsynthesis": ProbePlan(
            positive=ProbeCase(
                "synthesize a deterministic PD control action",
                lambda func: func(
                    np.array([1.0, -1.0], dtype=float),
                    np.array([0.0, 0.0], dtype=float),
                    0.0,
                ),
                _assert_control_synthesis,
            ),
            negative=ProbeCase(
                "reject a missing current state",
                lambda func: func(
                    None,
                    np.array([0.0, 0.0], dtype=float),
                    0.0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
