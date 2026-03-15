from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_hestonpathsampler


@register_atom(witness_hestonpathsampler)
@icontract.require(lambda S0: isinstance(S0, (float, int, np.number)) and S0 > 0, "S0 must be numeric and positive")
@icontract.require(lambda v0: isinstance(v0, (float, int, np.number)) and v0 > 0, "v0 must be numeric and positive")
@icontract.require(lambda rho: isinstance(rho, (float, int, np.number)) and -1 <= rho <= 1, "rho must be in [-1, 1]")
@icontract.require(lambda kappa: isinstance(kappa, (float, int, np.number)) and kappa > 0, "kappa must be positive")
@icontract.require(lambda theta: isinstance(theta, (float, int, np.number)) and theta > 0, "theta must be positive")
@icontract.require(lambda sigma_v: isinstance(sigma_v, (float, int, np.number)) and sigma_v > 0, "sigma_v must be positive")
@icontract.require(lambda T: isinstance(T, (float, int, np.number)) and T > 0, "T must be positive")
@icontract.require(lambda dt: isinstance(dt, (float, int, np.number)) and dt > 0, "dt must be positive")
@icontract.require(lambda num_sims: isinstance(num_sims, int) and num_sims >= 1, "num_sims must be an integer >= 1")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "must return a 2-tuple of arrays")
def hestonpathsampler(S0: float, v0: float, rho: float, kappa: float, theta: float, sigma_v: float, T: float, dt: float, num_sims: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate random stock price paths where volatility itself changes over time.

    Generates num_sims paths. Price and variance are driven by correlated
    random processes.

    Args:
        S0 — start price: initial stock price (> 0).
        v0: initial variance (> 0).
        rho: correlation between price and variance (-1 to 1).
        kappa: speed at which variance reverts to its long-run level (> 0).
        theta: long-run variance level (> 0).
        sigma_v: how much the variance itself fluctuates (> 0).
        T: total time horizon in years (> 0).
        dt: time step size (> 0, < T).
        num_sims: number of random paths to generate (>= 1).

    Returns:
        S_paths: all entries > 0; stock price paths from S0.
        v_paths: all entries >= 0; variance paths from v0.
    """
    N = int(T / dt)
    Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], (num_sims, N))
    Z_s, Z_v = Z[:, :, 0], Z[:, :, 1]
    S = np.zeros((num_sims, N))
    v = np.zeros((num_sims, N))
    S[:, 0] = S0
    v[:, 0] = v0
    for t in range(1, N):
        v[:, t] = np.maximum(v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + sigma_v * np.sqrt(v[:, t-1]) * np.sqrt(dt) * Z_v[:, t], 0)
        S[:, t] = S[:, t-1] * np.exp(-0.5 * v[:, t-1] * dt + np.sqrt(v[:, t-1]) * np.sqrt(dt) * Z_s[:, t])
    return S, v
