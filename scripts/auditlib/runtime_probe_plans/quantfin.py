"""Quantfin family runtime probe plans split from the monolithic registry."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_float_int_pair = rt._assert_float_int_pair
    _assert_float_list = rt._assert_float_list
    _assert_int_pair = rt._assert_int_pair
    _assert_scalar = rt._assert_scalar
    _assert_tuple = rt._assert_tuple
    safe_import_module = rt.safe_import_module

    def _assert_cashflow_list(expected: list[Any]) -> Callable[[Any], None]:
        def _check(value: Any) -> None:
            assert isinstance(value, list), f"expected list result, got {type(value)!r}"
            assert value == expected, f"expected {expected!r}, got {value!r}"

        return _check

    def _assert_complex_scalar(expected: complex) -> Callable[[Any], None]:
        def _check(value: Any) -> None:
            assert isinstance(value, complex), f"expected complex result, got {type(value)!r}"
            assert value == expected, f"expected {expected!r}, got {value!r}"

        return _check

    def _quick_sim_anti_positive(func: Callable[..., Any]) -> Any:
        module = safe_import_module("ageoa.quantfin.montecarlo")
        original = dict(module.SIMULATOR_REGISTRY)

        def _dummy_simulator(model: Any, claim: Any, rng: np.random.Generator, trials: int, anti: bool) -> float:
            _ = (model, claim, rng)
            return float(trials + (1 if anti else 3))

        try:
            module._register_simulator("unit_dummy", _dummy_simulator)
            model = safe_import_module("ageoa.quantfin.models").DiscretizeModel()
            claim = safe_import_module("ageoa.quantfin.models").ContingentClaim()
            return func(model, claim, 8, "unit_dummy")
        finally:
            module.SIMULATOR_REGISTRY.clear()
            module.SIMULATOR_REGISTRY.update(original)

    def _quick_sim_anti_negative(func: Callable[..., Any]) -> Any:
        module = safe_import_module("ageoa.quantfin.montecarlo")
        original = dict(module.SIMULATOR_REGISTRY)

        def _dummy_simulator(model: Any, claim: Any, rng: np.random.Generator, trials: int, anti: bool) -> float:
            _ = (model, claim, rng, anti)
            return float(trials)

        try:
            module._register_simulator("unit_dummy", _dummy_simulator)
            model = safe_import_module("ageoa.quantfin.models").DiscretizeModel()
            claim = safe_import_module("ageoa.quantfin.models").ContingentClaim()
            return func(model, claim, 7, "unit_dummy")
        finally:
            module.SIMULATOR_REGISTRY.clear()
            module.SIMULATOR_REGISTRY.update(original)

    return {
        "ageoa.quantfin.montecarlo.quick_sim_anti": ProbePlan(
            positive=ProbeCase(
                "run a deterministic antithetic Monte Carlo wrapper over a dummy seeded simulator",
                _quick_sim_anti_positive,
                _assert_scalar(6.0),
            ),
            negative=ProbeCase(
                "reject an odd trial count for antithetic Monte Carlo",
                _quick_sim_anti_negative,
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.local_vol_d12.allfort": ProbePlan(
            positive=ProbeCase(
                "extract one deterministic maturity slice from a simple quote surface",
                lambda func: func(
                    lambda callback, seq: [callback(item) for item in seq],
                    {(90.0, 1.0): 0.22, (100.0, 1.0): 0.2, (110.0, 1.0): 0.24},
                    [90.0, 100.0, 110.0],
                    1.0,
                    100.0,
                ),
                _assert_float_list([0.22, 0.2, 0.24]),
            ),
            negative=ProbeCase(
                "reject a non-dict quote surface",
                lambda func: func(
                    lambda callback, seq: [callback(item) for item in seq],
                    None,
                    [90.0, 100.0],
                    1.0,
                    100.0,
                ),
                expect_exception=True,
                ),
                parity_used=True,
            ),
        "ageoa.quantfin.local_vol_d12.var": ProbePlan(
            positive=ProbeCase(
                "compute implied variance from a deterministic volatility/time pair",
                lambda func: func(100.0, 1.0, 1.0, 0.2, {"surface": "flat"}),
                _assert_scalar(0.04000000000000001),
            ),
            negative=ProbeCase(
                "reject a non-positive strike for implied variance lookup",
                lambda func: func(0.0, 1.0, 1.0, 0.2, {"surface": "flat"}),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.local_vol_d12.localvol": ProbePlan(
            positive=ProbeCase(
                "compute Dupire local volatility for a deterministic denominator and derivative",
                lambda func: func(0.18, 100.0, 0.2, {"curve": "flat"}, 100.0, 0.09, np.sqrt, 1.0, 0.04, 0.04),
                _assert_scalar(np.sqrt(2.0)),
            ),
            negative=ProbeCase(
                "reject a non-positive current stock level",
                lambda func: func(0.18, 0.0, 0.2, {"curve": "flat"}, 100.0, 0.09, np.sqrt, 1.0, 0.04, 0.04),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.avg": ProbePlan(
            positive=ProbeCase(
                "compute the arithmetic mean over a short deterministic trial vector",
                lambda func: func(float, sum, 4, [1.0, 2.0, 3.0, 6.0]),
                _assert_scalar(3.0),
            ),
            negative=ProbeCase(
                "reject a non-positive trial count",
                lambda func: func(float, sum, 0, [1.0]),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.maxstep": ProbePlan(
            positive=ProbeCase(
                "return the default trading-day maximum time step",
                lambda func: func(),
                _assert_scalar(1.0 / 250.0),
            ),
            negative=ProbeCase(
                "reject unexpected positional arguments",
                lambda func: func(1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.insertcf": ProbePlan(
            positive=ProbeCase(
                "wrap a single deterministic cash flow in a singleton list",
                lambda func: func((1.25, 3.0)),
                _assert_cashflow_list([(1.25, 3.0)]),
            ),
            negative=ProbeCase(
                "reject a missing cash flow value",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.insertcflist": ProbePlan(
            positive=ProbeCase(
                "insert a short deterministic cash-flow list using a provided inserter",
                lambda func: func(
                    [(1.0, 10.0)],
                    lambda g: g,
                    lambda callback, init, seq: init,
                    lambda item, existing: existing + [item],
                    [(2.0, 5.0), (3.0, 7.0)],
                ),
                _assert_cashflow_list([(1.0, 10.0), (2.0, 5.0), (3.0, 7.0)]),
            ),
            negative=ProbeCase(
                "reject a non-list batch of cash flows",
                lambda func: func(
                    [(1.0, 10.0)],
                    lambda g: g,
                    lambda callback, init, seq: init,
                    lambda item, existing: existing + [item],
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.runmc": ProbePlan(
            positive=ProbeCase(
                "evaluate a deterministic Monte Carlo state pipeline",
                lambda func: func(
                    lambda state_value, init_state: state_value + init_state["base"],
                    lambda inner_value, rand_state: inner_value + rand_state["seed"],
                    lambda callback, value: callback(value),
                    {"base": 3.0},
                    lambda value: value + 1.0,
                    7.0,
                    {"seed": 2.0},
                    lambda lift, value: lift(value),
                ),
                _assert_scalar(13.0),
            ),
            negative=ProbeCase(
                "reject a missing initial state",
                lambda func: func(
                    lambda state_value, init_state: state_value + init_state["base"],
                    lambda inner_value, rand_state: inner_value + rand_state["seed"],
                    lambda callback, value: callback(value),
                    None,
                    lambda value: value + 1.0,
                    7.0,
                    {"seed": 2.0},
                    lambda lift, value: lift(value),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.runsimulation": ProbePlan(
            positive=ProbeCase(
                "run a deterministic simulation wrapper over a stubbed Monte Carlo runner",
                lambda func: func(
                    False,
                    {"claims": 1},
                    {"model": "unit"},
                    lambda seed, trials: seed["base"] + trials,
                    lambda run, seed, trials: run(seed, trials),
                    {"base": 4.0},
                    6,
                    0.0,
                ),
                _assert_scalar(10.0),
            ),
            negative=ProbeCase(
                "reject a non-positive trial count",
                lambda func: func(
                    False,
                    {"claims": 1},
                    {"model": "unit"},
                    lambda seed, trials: seed["base"] + trials,
                    lambda run, seed, trials: run(seed, trials),
                    {"base": 4.0},
                    0,
                    0.0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.runsimulationanti": ProbePlan(
            positive=ProbeCase(
                "average the normal and antithetic half-runs",
                lambda func: func(
                    {"claims": 1},
                    {"model": "unit"},
                    lambda modl, ccs, seed, half, anti: float(half + (10 if anti else 2)),
                    {"base": 4.0},
                    8,
                ),
                _assert_scalar(10.0),
            ),
            negative=ProbeCase(
                "reject a non-positive trial count",
                lambda func: func(
                    {"claims": 1},
                    {"model": "unit"},
                    lambda modl, ccs, seed, half, anti: float(half + (10 if anti else 2)),
                    {"base": 4.0},
                    0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.quicksim": ProbePlan(
            positive=ProbeCase(
                "run the quick simulation helper with a deterministic seed constructor",
                lambda func: func(
                    {"model": "unit"},
                    {"claims": 1},
                    lambda seed: {"base": float(seed)},
                    lambda mdl, opts, seed, trials, anti: seed["base"] + trials + (1.0 if anti else 0.0),
                    5,
                ),
                _assert_scalar(505.0),
            ),
            negative=ProbeCase(
                "reject a non-positive trial count",
                lambda func: func(
                    {"model": "unit"},
                    {"claims": 1},
                    lambda seed: {"base": float(seed)},
                    lambda mdl, opts, seed, trials, anti: seed["base"] + trials + (1.0 if anti else 0.0),
                    0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.quicksimanti": ProbePlan(
            positive=ProbeCase(
                "run the antithetic quick simulation helper with a deterministic seed constructor",
                lambda func: func(
                    {"model": "unit"},
                    {"claims": 1},
                    lambda seed: {"base": float(seed)},
                    lambda mdl, opts, seed, trials: seed["base"] + trials + 2.0,
                    5,
                ),
                _assert_scalar(507.0),
            ),
            negative=ProbeCase(
                "reject a non-positive trial count",
                lambda func: func(
                    {"model": "unit"},
                    {"claims": 1},
                    lambda seed: {"base": float(seed)},
                    lambda mdl, opts, seed, trials: seed["base"] + trials + 2.0,
                    0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.simulatestate": ProbePlan(
            positive=ProbeCase(
                "average a deterministic batch of trial results",
                lambda func: func(
                    False,
                    lambda trials, values: float(sum(values) / trials),
                    [],
                    {"model": "unit"},
                    lambda trials, single_trial: [single_trial] * trials,
                    2.5,
                    4,
                ),
                _assert_scalar(2.5),
            ),
            negative=ProbeCase(
                "reject a non-positive trial count",
                lambda func: func(
                    False,
                    lambda trials, values: float(sum(values) / trials),
                    [],
                    {"model": "unit"},
                    lambda trials, single_trial: [single_trial] * trials,
                    2.5,
                    0,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.monte_carlo_anti_d12.runsim": ProbePlan(
            positive=ProbeCase(
                "halve the trial count before dispatching the simulation variant",
                lambda func: func(
                    {"claims": 1},
                    lambda total, parts: total // parts,
                    {"model": "unit"},
                    lambda modl, ccs, seed, half_trials, anti: float(half_trials + (1 if anti else 0)),
                    {"seed": 5},
                    10,
                    True,
                ),
                _assert_scalar(6.0),
            ),
            negative=ProbeCase(
                "reject a non-positive trial count",
                lambda func: func(
                    {"claims": 1},
                    lambda total, parts: total // parts,
                    {"model": "unit"},
                    lambda modl, ccs, seed, half_trials, anti: float(half_trials + (1 if anti else 0)),
                    {"seed": 5},
                    0,
                    True,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.char_func_option_d12.f": ProbePlan(
            positive=ProbeCase(
                "evaluate the real-valued characteristic-function integrand at a simple deterministic point",
                lambda func: func(
                    np.exp,
                    1j,
                    complex(np.log(100.0), 0.0),
                    complex(2.0, 0.0),
                    lambda z: z.real,
                    complex(3.0, 0.0),
                    complex(0.0, 0.0),
                    complex(0.0, 0.0),
                ),
                _assert_scalar(6.0),
            ),
            negative=ProbeCase(
                "reject a non-complex log-strike argument",
                lambda func: func(
                    np.exp,
                    1j,
                    1.0,
                    complex(2.0, 0.0),
                    lambda z: z.real,
                    complex(3.0, 0.0),
                    complex(0.0, 0.0),
                    complex(0.0, 0.0),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.char_func_option_d12.cf": ProbePlan(
            positive=ProbeCase(
                "evaluate a deterministic martingale-corrected characteristic function at one complex frequency",
                lambda func: func(
                    lambda model, fg, tmat: lambda x: complex(model["scale"] * x.real + fg["offset"], x.imag + tmat),
                    {"offset": 0.25},
                    {"scale": 2.0},
                    1.5,
                    complex(0.5, -0.25),
                ),
                _assert_complex_scalar(complex(1.25, 1.25)),
            ),
            negative=ProbeCase(
                "reject a non-complex frequency argument",
                lambda func: func(
                    lambda model, fg, tmat: lambda x: complex(model["scale"] * x.real + fg["offset"], x.imag + tmat),
                    {"offset": 0.25},
                    {"scale": 2.0},
                    1.5,
                    0.5,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.char_func_option_d12.charfuncoption": ProbePlan(
            positive=ProbeCase(
                "price a deterministic option with zeroed inversion integrals and matching spot-strike terms",
                lambda func: func(
                    0.0,
                    lambda x: x,
                    lambda model, fg, tmat: lambda x: complex(x.real + model["shift"], x.imag + fg["shift"] + tmat),
                    0.95,
                    0.1,
                    0.2,
                    lambda t: 0.95,
                    np.exp,
                    lambda exp_fn, imag, k, left, real_part, right, v, v_prime: float(real_part(exp_fn(-imag * v * k) * left * right)),
                    {"shift": 0.0},
                    lambda v: 0.0,
                    lambda v: 0.0,
                    1j,
                    lambda func_inner, lower, upper, tol: 0.0,
                    float(np.log(100.0)),
                    complex(2.0, 0.0),
                    np.log,
                    {"shift": 0.0},
                    "call",
                    0.0,
                    0.0,
                    float(np.pi),
                    1.0,
                    lambda z: float(z.real),
                    complex(3.0, 0.0),
                    100.0,
                    100.0,
                    1.0,
                    complex(0.0, 0.0),
                    complex(0.0, 0.0),
                    0.0,
                    {"curve": "flat"},
                ),
                _assert_scalar(0.0),
            ),
            negative=ProbeCase(
                "reject a non-positive strike price",
                lambda func: func(
                    0.0,
                    lambda x: x,
                    lambda model, fg, tmat: lambda x: complex(x.real, x.imag),
                    0.95,
                    0.1,
                    0.2,
                    lambda t: 0.95,
                    np.exp,
                    lambda exp_fn, imag, k, left, real_part, right, v, v_prime: 0.0,
                    {"shift": 0.0},
                    lambda v: 0.0,
                    lambda v: 0.0,
                    1j,
                    lambda func_inner, lower, upper, tol: 0.0,
                    float(np.log(100.0)),
                    complex(2.0, 0.0),
                    np.log,
                    {"shift": 0.0},
                    "call",
                    0.0,
                    0.0,
                    float(np.pi),
                    1.0,
                    lambda z: float(z.real),
                    complex(3.0, 0.0),
                    100.0,
                    0.0,
                    1.0,
                    complex(0.0, 0.0),
                    complex(0.0, 0.0),
                    0.0,
                    {"curve": "flat"},
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.addmod64": ProbePlan(
            positive=ProbeCase(
                "compute a modular 64-bit sum deterministically",
                lambda func: func(7, 9, 10, lambda a, b, m: (a + b) % m),
                _assert_scalar(6),
            ),
            negative=ProbeCase(
                "reject a non-positive modulus",
                lambda func: func(7, 9, 0, lambda a, b, m: (a + b) % m),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.next": ProbePlan(
            positive=ProbeCase(
                "advance the generator one step and return the provided 64-bit word",
                lambda func: func(int, 11, 17, 23),
                _assert_int_pair(23, 17),
            ),
            negative=ProbeCase(
                "reject a negative generator state",
                lambda func: func(int, -1, 17, 23),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.randomdouble": ProbePlan(
            positive=ProbeCase(
                "convert a fixed 53-bit mantissa fragment into a uniform double",
                lambda func: func(divmod, float, 2**52, 11, 17),
                _assert_float_int_pair(0.5, 17),
            ),
            negative=ProbeCase(
                "reject a negative generator state",
                lambda func: func(divmod, float, 2**52, -1, 17),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.randomint": ProbePlan(
            positive=ProbeCase(
                "convert a fixed 64-bit word to a platform integer",
                lambda func: func(int, 11, 17, 23),
                _assert_int_pair(23, 17),
            ),
            negative=ProbeCase(
                "reject a negative generator state",
                lambda func: func(int, -1, 17, 23),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.randomint64": ProbePlan(
            positive=ProbeCase(
                "convert a fixed 64-bit word to a signed 64-bit integer",
                lambda func: func(int, 11, 17, 23),
                _assert_int_pair(23, 17),
            ),
            negative=ProbeCase(
                "reject a negative generator state",
                lambda func: func(int, -1, 17, 23),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.randomword32": ProbePlan(
            positive=ProbeCase(
                "generate a deterministic 32-bit word and updated state",
                lambda func: func(5, 11, 17, 9, lambda a, b: a ^ b),
                _assert_int_pair((5 ^ 9) & 0xFFFFFFFF, 17),
            ),
            negative=ProbeCase(
                "reject a negative generator state",
                lambda func: func(5, -1, 17, 9, lambda a, b: a ^ b),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.randomword64": ProbePlan(
            positive=ProbeCase(
                "join two deterministic 32-bit words into a 64-bit word",
                lambda func: func(lambda y1, y2: (y1 << 32) | y2, 11, 17, 1, 2),
                _assert_int_pair((1 << 32) | 2, 17),
            ),
            negative=ProbeCase(
                "reject a negative generator state",
                lambda func: func(lambda y1, y2: (y1 << 32) | y2, -1, 17, 1, 2),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.mulmod64": ProbePlan(
            positive=ProbeCase(
                "compute a modular 64-bit product deterministically",
                lambda func: func(7, 9, lambda *args: 0, 10),
                _assert_scalar(3),
            ),
            negative=ProbeCase(
                "reject a non-positive modulus",
                lambda func: func(7, 9, lambda *args: 0, 0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.powmod64": ProbePlan(
            positive=ProbeCase(
                "compute modular exponentiation deterministically",
                lambda func: func(7, 5, lambda *args: 0, 11),
                _assert_scalar(10),
            ),
            negative=ProbeCase(
                "reject a negative exponent",
                lambda func: func(7, -1, lambda *args: 0, 11),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.skip": ProbePlan(
            positive=ProbeCase(
                "return the supplied advanced generator state",
                lambda func: func(5, 11, 42),
                _assert_scalar(42),
            ),
            negative=ProbeCase(
                "reject a negative skip distance",
                lambda func: func(-1, 11, 42),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.rng_skip_d12.split": ProbePlan(
            positive=ProbeCase(
                "split a generator into skipped and original streams",
                lambda func: func(11, lambda d, st: st + d, 7),
                _assert_tuple((18, 11)),
            ),
            negative=ProbeCase(
                "reject a negative generator state",
                lambda func: func(-1, lambda d, st: st + d, 7),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.tdma_solver_d12.tdmasolver": ProbePlan(
            positive=ProbeCase(
                "solve a simple three-by-three tridiagonal system with the Thomas algorithm",
                lambda func: func(
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    0.0,
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    2.0,
                    [-1.0, -1.0, 0.0],
                    [],
                    [-1.0, -1.0, 0.0],
                    [],
                    -1.0,
                    -1.0,
                    -1.0,
                    [1.0, 0.0, 1.0],
                    [],
                    [1.0, 0.0, 1.0],
                    [],
                    1.0,
                    1.0,
                    lambda *args, **kwargs: None,
                    list,
                    lambda xs: xs[0],
                    lambda xs: xs[-1],
                    len,
                    map,
                    lambda n: [0.0] * n,
                    lambda xs, i: xs[i],
                    lambda xs: list(reversed(xs)),
                    lambda thunk: thunk(),
                    lambda xs: list(xs),
                    list,
                    lambda xs: xs,
                    lambda xs, i, value: xs.__setitem__(i, value),
                    [],
                    0.0,
                    [],
                ),
                _assert_float_list([1.0, 1.0, 1.0]),
            ),
            negative=ProbeCase(
                "reject an empty main diagonal",
                lambda func: func(
                    [0.0],
                    [0.0],
                    0.0,
                    [],
                    [],
                    0.0,
                    [0.0],
                    [],
                    [0.0],
                    [],
                    0.0,
                    0.0,
                    0.0,
                    [1.0],
                    [],
                    [1.0],
                    [],
                    1.0,
                    1.0,
                    lambda *args, **kwargs: None,
                    list,
                    lambda xs: xs[0],
                    lambda xs: xs[-1],
                    len,
                    map,
                    lambda n: [0.0] * n,
                    lambda xs, i: xs[i],
                    lambda xs: list(reversed(xs)),
                    lambda thunk: thunk(),
                    lambda xs: list(xs),
                    list,
                    lambda xs: xs,
                    lambda xs, i, value: xs.__setitem__(i, value),
                    [],
                    0.0,
                    [],
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quantfin.tdma_solver_d12.cotraversevec": ProbePlan(
            positive=ProbeCase(
                "aggregate aligned entries across a pair of numeric vectors",
                lambda func: func(
                    lambda start, length: list(range(start, length)),
                    lambda projected: float(sum(projected)),
                    lambda projector, wrapped: [projector(vec) for vec in wrapped],
                    0,
                    3,
                    [
                        [1.0, 2.0, 3.0],
                        [10.0, 20.0, 30.0],
                    ],
                    lambda mapper, indices: [mapper(idx) for idx in indices],
                ),
                _assert_float_list([11.0, 22.0, 33.0]),
            ),
            negative=ProbeCase(
                "reject a non-positive output length",
                lambda func: func(
                    lambda start, length: list(range(start, length)),
                    lambda projected: float(sum(projected)),
                    lambda projector, wrapped: [projector(vec) for vec in wrapped],
                    0,
                    0,
                    [[1.0, 2.0]],
                    lambda mapper, indices: [mapper(idx) for idx in indices],
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
