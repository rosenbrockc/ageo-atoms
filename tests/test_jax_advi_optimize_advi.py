import inspect

import jax.numpy as jnp

from ageoa.jax_advi.optimize_advi.atoms import meanfieldvariationalfit


def _log_prior(theta):
    return -0.5 * jnp.sum(theta["theta"] ** 2)


def _log_lik(theta):
    return -0.5 * jnp.sum((theta["theta"] - 1.0) ** 2)


def test_meanfieldvariationalfit_matches_upstream_optional_defaults() -> None:
    signature = inspect.signature(meanfieldvariationalfit)
    assert signature.parameters["M"].default == 100
    assert signature.parameters["verbose"].default is False
    assert signature.parameters["seed"].default == 2
    assert signature.parameters["n_draws"].default == 1000
    assert signature.parameters["opt_method"].default == "trust-ncg"


def test_meanfieldvariationalfit_returns_latent_state_and_objective() -> None:
    free_means, free_sds, objective_fun, rng_state_out = meanfieldvariationalfit(
        {"theta": (1,)},
        _log_prior,
        _log_lik,
        M=2,
        verbose=False,
        seed=2,
        n_draws=4,
        var_param_inits={"mean": (0.0, 0.01), "log_sd": (-1.0, 0.01)},
        opt_method="L-BFGS-B",
    )

    assert set(free_means) == {"theta"}
    assert set(free_sds) == {"theta"}
    assert tuple(free_means["theta"].shape) == (1,)
    assert tuple(free_sds["theta"].shape) == (1,)
    assert bool((free_sds["theta"] > 0).all())
    assert callable(objective_fun)
    assert rng_state_out == 3
