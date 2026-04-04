from ageoa.quantfin.monte_carlo_anti_d12.atoms import evolve, quicksim, runsimulationanti


def test_quicksim_uses_default_seed_constructor() -> None:
    result = quicksim(
        {"model": "unit"},
        {"claims": 1},
        lambda seed: {"base": float(seed)},
        lambda mdl, opts, seed, trials, anti: seed["base"] + trials + (1.0 if anti else 0.0),
        5,
    )

    assert result == 505.0


def test_runsimulationanti_averages_normal_and_antithetic_halves() -> None:
    result = runsimulationanti(
        {"claims": 1},
        {"model": "unit"},
        lambda modl, ccs, seed, half, anti: float(half + (10 if anti else 2)),
        {"base": 4.0},
        8,
    )

    assert result == 10.0


def test_evolve_recurses_using_full_signature() -> None:
    result = evolve(
        False,
        evolve,
        lambda mdl, anti, start, stop: {"model": mdl["name"], "anti": anti, "start": start, "stop": stop},
        lambda: {"model": "unit", "anti": False, "start": 0.0, "stop": 0.0},
        lambda mdl: 1.0,
        {"name": "unit"},
        1.0,
        0.0,
        2.5,
        lambda t2, t1: t2 - t1,
        lambda t, offset: t + offset,
        lambda cond, thunk: thunk() if not cond else None,
    )

    assert result == {"model": "unit", "anti": False, "start": 2.0, "stop": 2.5}
