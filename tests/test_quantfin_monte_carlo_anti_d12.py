from ageoa.quantfin.monte_carlo_anti_d12.atoms import quicksim, runsimulationanti


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
