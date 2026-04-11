"""Runtime probe plans for scipy.stats and scipy.integrate families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_scalar = rt._assert_scalar

    return {
        "scipy.stats.ttest_ind": ProbePlan(
            positive=ProbeCase(
                "ttest_ind over two distinct samples",
                lambda func: func(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
                lambda result: np.testing.assert_allclose(
                    np.array([result.statistic, result.pvalue]),
                    np.array([-3.6742346141747673, 0.021311641128756727]),
                    atol=1e-8,
                ),
            ),
            negative=ProbeCase(
                "ttest_ind rejects None input",
                lambda func: func(None, np.array([1.0, 2.0])),
                expect_exception=True,
            ),
        ),
        "scipy.stats.pearsonr": ProbePlan(
            positive=ProbeCase(
                "pearsonr over perfectly correlated samples",
                lambda func: func(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])),
                lambda result: np.testing.assert_allclose(
                    np.array([result.statistic, result.pvalue]),
                    np.array([1.0, 0.0]),
                    atol=1e-12,
                ),
            ),
            negative=ProbeCase(
                "pearsonr rejects too-short input",
                lambda func: func(np.array([1.0]), np.array([1.0])),
                expect_exception=True,
            ),
        ),
        "scipy.stats.norm": ProbePlan(
            positive=ProbeCase(
                "norm returns a frozen normal distribution",
                lambda func: func(loc=1.0, scale=2.0),
                lambda result: np.testing.assert_allclose(
                    np.array([result.mean(), result.std()]),
                    np.array([1.0, 2.0]),
                    atol=1e-12,
                ),
            ),
            negative=ProbeCase(
                "norm rejects non-positive scale",
                lambda func: func(scale=0.0),
                expect_exception=True,
            ),
        ),
        "scipy.integrate.quad": ProbePlan(
            positive=ProbeCase(
                "quad integrates x^2 from 0 to 1",
                lambda func: func(lambda x: x * x, 0.0, 1.0),
                lambda result: np.testing.assert_allclose(
                    np.array(result[:2]),
                    np.array([1.0 / 3.0, result[1]]),
                    atol=1e-8,
                ),
            ),
            negative=ProbeCase(
                "quad rejects a missing function",
                lambda func: func(None, 0.0, 1.0),
                expect_exception=True,
            ),
        ),
        "scipy.integrate.simpson": ProbePlan(
            positive=ProbeCase(
                "simpson integrates a quadratic sample",
                lambda func: func(np.array([0.0, 1.0, 4.0]), x=np.array([0.0, 1.0, 2.0])),
                _assert_scalar(8.0 / 3.0),
            ),
            negative=ProbeCase(
                "simpson rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "scipy.integrate.solve_ivp": ProbePlan(
            positive=ProbeCase(
                "solve_ivp integrates y'=-y over a short interval",
                lambda func: func(
                    lambda t, y: -y,
                    (0.0, 1.0),
                    np.array([1.0]),
                    t_eval=np.array([0.0, 1.0]),
                ),
                lambda result: np.testing.assert_allclose(
                    result.y[:, -1], np.array([np.exp(-1.0)]), atol=5e-3
                ),
            ),
            negative=ProbeCase(
                "solve_ivp rejects missing initial condition",
                lambda func: func(lambda t, y: -y, (0.0, 1.0), None),
                expect_exception=True,
            ),
        ),
    }
