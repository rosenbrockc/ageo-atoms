"""Runtime probe plans for scipy.optimize_v2 families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_optimize_result_near = rt._assert_optimize_result_near

    def _quadratic(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float((x[0] - 1.0) ** 2)

    return {
        "ageoa.scipy.optimize_v2.differentialevolutionoptimization": ProbePlan(
            positive=ProbeCase(
                "Differential evolution minimizes a one-dimensional quadratic on bounded input",
                lambda func: func(
                    _quadratic,
                    [(0.0, 2.0)],
                    maxiter=24,
                    popsize=8,
                    tol=0.0,
                    atol=0.0,
                    rng=np.random.default_rng(7),
                    workers=1,
                    polish=True,
                ),
                _assert_optimize_result_near(1.0, atol=1e-1),
            ),
            negative=ProbeCase(
                "reject malformed bounds",
                lambda func: func(_quadratic, [(0.0,)], rng=np.random.default_rng(7)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.optimize_v2.shgoglobaloptimization": ProbePlan(
            positive=ProbeCase(
                "SHGO minimizes a one-dimensional quadratic on a bounded interval",
                lambda func: func(
                    _quadratic,
                    [(0.0, 2.0)],
                    (),
                    (),
                    16,
                    1,
                    None,
                    {},
                    {},
                    "simplicial",
                ),
                _assert_optimize_result_near(1.0),
            ),
            negative=ProbeCase(
                "reject malformed bounds",
                lambda func: func(_quadratic, [(0.0,)], (), (), 16, 1, None, {}, {}, "simplicial"),
                expect_exception=True,
            ),
        ),
    }
