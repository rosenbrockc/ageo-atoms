"""Runtime probe plans for conjugate priors and the remaining small MCMC helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_value = rt._assert_value

    target_log = lambda x: float(-0.5 * np.dot(x, x))
    mala_mean = lambda x: 2.0 * np.asarray(x, dtype=float)

    def _assert_de_kernel(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        state, rng = result
        assert np.asarray(state).shape == (3, 2)
        assert np.asarray(rng).shape == (2,)

    return {
        "ageoa.conjugate_priors.beta_binom.posterior_randmodel": ProbePlan(
            positive=ProbeCase(
                "compute a Beta-Binomial posterior update from binary observations",
                lambda func: func(
                    np.array([2.0, 3.0], dtype=float),
                    np.eye(2),
                    np.array([1.0, 0.0, 1.0, 1.0], dtype=float),
                ),
                _assert_array(np.array([5.0, 4.0], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing prior vector",
                lambda func: func(None, np.eye(2), np.array([1.0, 0.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.conjugate_priors.beta_binom.posterior_randmodel_weighted": ProbePlan(
            positive=ProbeCase(
                "compute a weighted Beta-Binomial posterior update",
                lambda func: func(
                    np.array([2.0, 3.0], dtype=float),
                    np.eye(2),
                    np.array([1.0, 0.0, 1.0], dtype=float),
                    np.array([1.0, 0.5, 2.0], dtype=float),
                ),
                _assert_array(np.array([5.0, 3.5], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing weight vector",
                lambda func: func(
                    np.array([2.0, 3.0], dtype=float),
                    np.eye(2),
                    np.array([1.0, 0.0, 1.0], dtype=float),
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.conjugate_priors.normal.normal_gamma_posterior_update": ProbePlan(
            positive=ProbeCase(
                "compute a Normal-Gamma posterior update from sufficient statistics",
                lambda func: func(
                    {"mu0": 0.0, "kappa0": 1.0, "alpha0": 2.0, "beta0": 3.0},
                    {"n": 4.0, "mean": 1.5, "var": 2.0},
                ),
                _assert_value({"mu0": 1.2, "kappa0": 5.0, "alpha0": 4.0, "beta0": 7.9}),
            ),
            negative=ProbeCase(
                "reject missing sufficient statistics",
                lambda func: func({"mu0": 0.0, "kappa0": 1.0, "alpha0": 2.0, "beta0": 3.0}, None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel": ProbePlan(
            positive=ProbeCase(
                "build and run one Differential Evolution transition kernel on a small population",
                lambda func: func(target_log)(
                    np.array([[0.0, 0.5], [1.0, -0.5], [-0.25, 0.75]], dtype=float),
                    np.array([3, 5], dtype=np.int64),
                ),
                _assert_de_kernel,
            ),
            negative=ProbeCase(
                "reject a missing target log-kernel oracle",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.mala.mala_proposal_adjustment": ProbePlan(
            positive=ProbeCase(
                "compute the deterministic MALA proposal adjustment term",
                lambda func: func(0.5, np.array([1.0, -1.0], dtype=float), mala_mean),
                _assert_array(np.array([1.25, -1.25], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a non-numeric step size",
                lambda func: func("bad", np.array([1.0, -1.0], dtype=float), mala_mean),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
