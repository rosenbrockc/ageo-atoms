"""Runtime probe plans for rust_robotics families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan

    def _assert_bicycle_dynamics_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        x_dot, jacobian, u_inferred = result
        np.testing.assert_allclose(np.asarray(x_dot, dtype=float), np.array([2.0, 0.0, 0.0, 0.5]))
        assert np.asarray(jacobian, dtype=float).shape == (4, 4)
        np.testing.assert_allclose(np.asarray(u_inferred, dtype=float), np.array([0.0, 0.5]))

    return {
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
    }
