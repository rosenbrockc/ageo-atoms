"""MCMC foundational family runtime probe plans split from the monolithic registry."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_scalar = rt._assert_scalar
    safe_import_module = rt.safe_import_module

    def _mcmc_foundational_plans() -> dict[str, ProbePlan]:
        target_log = lambda x: float(-0.5 * np.dot(x, x))
        tensor_fn = lambda x: np.eye(x.shape[0], dtype=float)

        def _assert_mh_tuple(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            state, rng = result
            assert np.asarray(state).shape == (2,)
            assert np.asarray(rng).shape == (2,)

        def _assert_hmc_init(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            kernel_spec, chain_state = result
            np.testing.assert_allclose(np.asarray(kernel_spec, dtype=float), np.array([0.1, 4.0, 2.0], dtype=float))
            assert np.asarray(chain_state).shape == (5,)

        def _assert_hmc_rng(result: Any) -> None:
            arr = np.asarray(result)
            assert arr.shape == (1,)
            assert np.issubdtype(arr.dtype, np.integer)
            np.testing.assert_array_equal(arr, np.array([7], dtype=np.int64))

        def _assert_nuts_init(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            nuts_state, rng_key = result
            assert np.asarray(nuts_state).shape == (5,)
            np.testing.assert_array_equal(np.asarray(rng_key), np.array([7], dtype=np.int64))

        def _assert_mini_hmc_init(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            chain_state, kernel_static = result
            assert np.asarray(chain_state).shape == (4,)
            np.testing.assert_allclose(np.asarray(kernel_static, dtype=float), np.array([0.1, 4.0, 1.0], dtype=float))

        def _assert_hmc_transition(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 3
            state_out, prng_key_out, stats = result
            assert np.asarray(state_out).shape == (5,)
            assert np.asarray(prng_key_out).shape == (1,)
            assert isinstance(stats, dict)
            assert {"accepted", "accept_prob", "delta_H"} <= set(stats)

        def _assert_nuts_transition(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 4
            samples, trace, nuts_state_out, rng_key_out = result
            assert np.asarray(samples).shape == (2, 1)
            assert np.asarray(trace).shape == (3, 3)
            assert np.asarray(nuts_state_out).shape == (5,)
            assert np.asarray(rng_key_out).shape == (1,)

        def _assert_mini_hmc_proposal(result: Any) -> None:
            arr = np.asarray(result, dtype=float)
            assert arr.shape == (4,)
            assert np.all(np.isfinite(arr))

        def _assert_mini_hmc_transition(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            chain_state_out, transition_stats = result
            state = np.asarray(chain_state_out, dtype=float)
            stats = np.asarray(transition_stats, dtype=float)
            assert state.shape == (4,)
            assert stats.shape == (3,)
            assert np.all(np.isfinite(stats))
            assert 0.0 <= stats[1] <= 1.0

        def _assert_sampling_loop(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 3
            samples, trace, hmc_state_out = result
            assert np.asarray(samples).shape == (2, 1)
            assert np.asarray(trace).shape == (3, 3)
            assert np.asarray(hmc_state_out).shape == (4,)

        def _assert_nuts_tree(result: Any) -> None:
            arr = np.asarray(result, dtype=float)
            assert arr.shape == (1,)
            assert np.all(np.isfinite(arr))

        def _assert_dispatch_draws(result: Any) -> None:
            arr = np.asarray(result, dtype=float)
            assert arr.shape == (4, 2)
            assert np.all(np.isfinite(arr))

        def _assert_advancedhmc_tempering(result: Any) -> None:
            assert np.isclose(float(result), 1.0)

        def _assert_advancedhmc_transition(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            z_out, is_valid = result
            arr = np.asarray(z_out, dtype=float)
            assert arr.shape == (2,)
            assert np.allclose(arr, np.array([0.65, -0.6], dtype=float))
            assert is_valid is True

        def _invoke_rwmh(func: Callable[..., Any]) -> Any:
            kernel = func(target_log)
            return kernel(np.array([0.5, -0.5], dtype=float), np.array([1, 2], dtype=np.int64))

        def _invoke_rmhmc(func: Callable[..., Any]) -> Any:
            kernel = func(target_log, tensor_fn)
            return kernel(np.array([0.5, -0.5], dtype=float), np.array([1, 2], dtype=np.int64))

        def _invoke_hmc_builder(func: Callable[..., Any]) -> Any:
            kernel = func(target_log)
            return kernel(np.array([0.5, -0.5], dtype=float), np.array([1, 2], dtype=np.int64))

        def _invoke_nuts(func: Callable[..., Any]) -> Any:
            module = safe_import_module("ageoa.mcmc_foundational.mini_mcmc.nuts_llm.atoms")
            nuts_state, rng_key = module.initializenutsstate(target_log, 0.2, 0.8, 7)
            return func(nuts_state, rng_key, 2, 1)

        return {
            "ageoa.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel": ProbePlan(
                positive=ProbeCase(
                    "run one deterministic AEES Metropolis-Hastings step with an explicit RNG state",
                    lambda func: func(1.0, target_log, np.array([1, 2], dtype=np.int64)),
                    _assert_mh_tuple,
                ),
                negative=ProbeCase(
                    "reject a non-numeric tempering value",
                    lambda func: func("bad", target_log, np.array([1, 2], dtype=np.int64)),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle": ProbePlan(
                positive=ProbeCase(
                    "evaluate the tempered log-kernel for a candidate state",
                    lambda func: func(np.array([1.0, -1.0], dtype=float), 0.5),
                    _assert_scalar(0.0),
                ),
                negative=ProbeCase(
                    "reject a non-numeric tempering value",
                    lambda func: func(np.array([1.0, -1.0], dtype=float), "bad"),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.kthohr_mcmc.hmc.buildhmckernelfromlogdensityoracle": ProbePlan(
                positive=ProbeCase(
                    "build and run one HMC transition kernel from a target log-density oracle",
                    _invoke_hmc_builder,
                    _assert_mh_tuple,
                ),
                negative=ProbeCase(
                    "reject a missing log-density oracle",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel": ProbePlan(
                positive=ProbeCase(
                    "build and run one RMHMC transition kernel from oracle and tensor functions",
                    _invoke_rmhmc,
                    _assert_mh_tuple,
                ),
                negative=ProbeCase(
                    "reject a missing tensor oracle",
                    lambda func: func(target_log, None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.kthohr_mcmc.rwmh.constructrandomwalkmetropoliskernel": ProbePlan(
                positive=ProbeCase(
                    "build and run one random-walk Metropolis kernel",
                    _invoke_rwmh,
                    _assert_mh_tuple,
                ),
                negative=ProbeCase(
                    "reject a missing target log-kernel oracle",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.kthohr_mcmc.mcmc_algos.dispatch_mcmc_algorithm": ProbePlan(
                positive=ProbeCase(
                    "dispatch a deterministic random-walk sampler loop into posterior draws",
                    lambda func: func(
                        np.array([0.5, -0.25], dtype=float),
                        np.array([1.0, -1.0], dtype=float),
                        4,
                    ),
                    _assert_dispatch_draws,
                ),
                negative=ProbeCase(
                    "reject a missing initial state",
                    lambda func: func(np.array([0.5, -0.25], dtype=float), None, 4),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializehmckernelstate": ProbePlan(
                positive=ProbeCase(
                    "initialize a deterministic HMC kernel spec and chain state",
                    lambda func: func(target_log, np.array([0.5, -0.5], dtype=float), 0.1, 4),
                    _assert_hmc_init,
                ),
                negative=ProbeCase(
                    "reject a non-numeric step size",
                    lambda func: func(target_log, np.array([0.5, -0.5], dtype=float), "bad", 4),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializesamplerrng": ProbePlan(
                positive=ProbeCase(
                    "initialize a deterministic mini-mcmc sampler RNG key",
                    lambda func: func(7),
                    _assert_hmc_rng,
                ),
                negative=ProbeCase(
                    "reject a non-integer sampler seed",
                    lambda func: func(7.5),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.hamiltoniantransitionkernel": ProbePlan(
                positive=ProbeCase(
                    "run one seeded HMC transition from an explicit chain state and kernel spec",
                    lambda func: func(
                        np.array([0.5, -0.5, -0.25, -0.5, 0.5], dtype=float),
                        np.array([0.1, 4.0, 2.0], dtype=float),
                        np.array([7], dtype=np.int64),
                        target_log,
                    ),
                    _assert_hmc_transition,
                ),
                negative=ProbeCase(
                    "reject a missing kernel specification",
                    lambda func: func(
                        np.array([0.5, -0.5, -0.25, -0.5, 0.5], dtype=float),
                        None,
                        np.array([7], dtype=np.int64),
                        target_log,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.hmc.initializehmcstate": ProbePlan(
                positive=ProbeCase(
                    "initialize deterministic mini-mcmc HMC state and static kernel parameters",
                    lambda func: func(target_log, np.array([0.5], dtype=float), 0.1, 4, 7),
                    _assert_mini_hmc_init,
                ),
                negative=ProbeCase(
                    "reject a non-numeric step size for mini-mcmc HMC initialization",
                    lambda func: func(target_log, np.array([0.5], dtype=float), "bad", 4, 7),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.hmc.leapfrogproposalkernel": ProbePlan(
                positive=ProbeCase(
                    "run one deterministic leapfrog proposal step in mini-mcmc HMC",
                    lambda func: func(
                        np.array([0.5, -0.125, 0.0], dtype=float),
                        np.array([0.1, 2.0, 1.0], dtype=float),
                        target_log,
                    ),
                    _assert_mini_hmc_proposal,
                ),
                negative=ProbeCase(
                    "reject a missing kernel specification for mini-mcmc leapfrog",
                    lambda func: func(np.array([0.5, -0.125, 0.0], dtype=float), None, target_log),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition": ProbePlan(
                positive=ProbeCase(
                    "run one deterministic mini-mcmc HMC Metropolis transition",
                    lambda func: func(
                        np.array([0.5, -0.125, 0.0, 7.0], dtype=float),
                        np.array([0.1, 2.0, 1.0], dtype=float),
                        np.array([0.51, -0.13, -0.2601, -0.50995, 0.049495, -0.05049995, -0.00499975], dtype=float),
                    ),
                    _assert_mini_hmc_transition,
                ),
                negative=ProbeCase(
                    "reject a missing proposal state for mini-mcmc HMC transition",
                    lambda func: func(np.array([0.5, -0.125, 0.0, 7.0], dtype=float), np.array([0.1, 2.0, 1.0], dtype=float), None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.hmc.runsamplingloop": ProbePlan(
                positive=ProbeCase(
                    "run a tiny deterministic mini-mcmc HMC sampling loop",
                    lambda func: func(np.array([0.5, -0.125, 0.0, 7.0], dtype=float), 2, 1),
                    _assert_sampling_loop,
                ),
                negative=ProbeCase(
                    "reject a missing initial mini-mcmc HMC state",
                    lambda func: func(None, 2, 1),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build": ProbePlan(
                positive=ProbeCase(
                    "build a deterministic shallow NUTS subtree in mini-mcmc",
                    lambda func: func(
                        1,
                        0.1,
                        -1.0,
                        np.array([0.5], dtype=float),
                        target_log,
                        lambda state, step_size, direction: np.asarray(state, dtype=float) + direction * step_size,
                        1,
                    ),
                    _assert_nuts_tree,
                ),
                negative=ProbeCase(
                    "reject a non-numeric step size for mini-mcmc NUTS tree build",
                    lambda func: func(1, "bad", -1.0, np.array([0.5], dtype=float), target_log, lambda state, step_size, direction: state, 1),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.advancedhmc.integrator.temperingfactorcomputation": ProbePlan(
                positive=ProbeCase(
                    "compute a deterministic AdvancedHMC tempering factor at the midpoint step",
                    lambda func: func(
                        np.array([1.0, -1.0], dtype=float),
                        np.array([0.5, 0.25], dtype=float),
                        2,
                        4,
                    ),
                    _assert_advancedhmc_tempering,
                ),
                negative=ProbeCase(
                    "reject a missing AdvancedHMC step count",
                    lambda func: func(
                        np.array([1.0, -1.0], dtype=float),
                        np.array([0.5, 0.25], dtype=float),
                        2,
                        None,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.advancedhmc.integrator.hamiltonianphasepointtransition": ProbePlan(
                positive=ProbeCase(
                    "run one deterministic AdvancedHMC phase-point transition",
                    lambda func: func(
                        np.array([1.0, 2.0], dtype=float),
                        np.array([0.1, -0.2], dtype=float),
                        np.array([0.5, 0.0], dtype=float),
                        1.5,
                    ),
                    _assert_advancedhmc_transition,
                ),
                negative=ProbeCase(
                    "reject a missing AdvancedHMC tempering scale",
                    lambda func: func(
                        np.array([1.0, 2.0], dtype=float),
                        np.array([0.1, -0.2], dtype=float),
                        np.array([0.5, 0.0], dtype=float),
                        None,
                    ),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.initializenutsstate": ProbePlan(
                positive=ProbeCase(
                    "initialize deterministic mini-mcmc NUTS state and RNG key",
                    lambda func: func(target_log, 0.2, 0.8, 7),
                    _assert_nuts_init,
                ),
                negative=ProbeCase(
                    "reject a non-numeric target acceptance probability",
                    lambda func: func(target_log, 0.2, "bad", 7),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions": ProbePlan(
                positive=ProbeCase(
                    "run a small seeded NUTS transition loop with explicit discard and collection counts",
                    _invoke_nuts,
                    _assert_nuts_transition,
                ),
                negative=ProbeCase(
                    "reject a missing RNG key",
                    lambda func: func(np.zeros(7, dtype=float), None, 2, 1),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
        }

    return _mcmc_foundational_plans()
