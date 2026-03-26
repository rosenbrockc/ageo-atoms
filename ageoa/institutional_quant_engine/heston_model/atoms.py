from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_simulate_heston_paths


@register_atom(witness_simulate_heston_paths)
@icontract.require(lambda S0: S0 is not None, "S0 cannot be None")
@icontract.require(lambda v0: v0 is not None, "v0 cannot be None")
@icontract.require(lambda kappa: kappa is not None, "kappa cannot be None")
@icontract.require(lambda theta: theta is not None, "theta cannot be None")
@icontract.require(lambda sigma_v: sigma_v is not None, "sigma_v cannot be None")
@icontract.require(lambda rho: rho is not None, "rho cannot be None")
@icontract.require(lambda n_steps: n_steps is not None, "n_steps cannot be None")
@icontract.require(lambda n_sims: n_sims is not None, "n_sims cannot be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def simulate_heston_paths(S0: float, v0: float, kappa: float, theta: float, sigma_v: float, rho: float, n_steps: int, n_sims: int) -> np.ndarray:
    """Generates random asset price paths where price volatility itself changes over time.

    Args:
        S0: initial asset price
        v0: initial variance
        kappa: speed at which variance returns to its long-run level
        theta: long-run variance level
        sigma_v: how much the variance itself fluctuates
        rho: correlation between price and variance (-1 to 1)
        n_steps: number of time steps per path
        n_sims: number of random paths to generate

    Returns:
        Simulated price paths, shape (n_sims, n_steps)
    """
    dt = 1.0 / n_steps
    Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], (n_sims, n_steps))
    Z_s, Z_v = Z[:, :, 0], Z[:, :, 1]
    S = np.zeros((n_sims, n_steps))
    v = np.zeros((n_sims, n_steps))
    S[:, 0] = S0
    v[:, 0] = v0
    for t in range(1, n_steps):
        v[:, t] = np.maximum(v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + sigma_v * np.sqrt(v[:, t-1]) * np.sqrt(dt) * Z_v[:, t], 0)
        S[:, t] = S[:, t-1] * np.exp(-0.5 * v[:, t-1] * dt + np.sqrt(v[:, t-1]) * np.sqrt(dt) * Z_s[:, t])
    return S