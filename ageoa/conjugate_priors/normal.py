from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .normal_witnesses import witness_normal_gamma_posterior_update

# juliacall unavailable; reimplemented in pure numpy


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_normal_gamma_posterior_update)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda prior: prior is not None, "prior cannot be None")
@icontract.require(lambda ss: ss is not None, "ss cannot be None")
@icontract.ensure(lambda result: result is not None, "normal_gamma_posterior_update output must not be None")
def normal_gamma_posterior_update(prior: dict[str, float] | tuple[float, float, float, float], ss: dict[str, float] | tuple[float, float, float]) -> dict[str, float]:
    """Computes a closed-form Normal-Gamma posterior from a Normal-Gamma prior and sufficient statistics as a pure, immutable conjugate update.

    Args:
        prior: Valid Normal-Gamma parameters (e.g., positive precision/shape/scale terms).
        ss: Contains the moments/count terms required by the conjugate posterior formula.

    Returns:
        Returned as a new immutable object; input prior is not mutated.
    """
    # Normal-Gamma conjugate update
    # prior: {mu0, kappa0, alpha0, beta0}
    # ss: {n, mean, var} (sufficient statistics)
    mu0 = prior.get('mu0', prior.get('mu', 0.0)) if isinstance(prior, dict) else prior[0]
    kappa0 = prior.get('kappa0', prior.get('kappa', 1.0)) if isinstance(prior, dict) else prior[1]
    alpha0 = prior.get('alpha0', prior.get('alpha', 1.0)) if isinstance(prior, dict) else prior[2]
    beta0 = prior.get('beta0', prior.get('beta', 1.0)) if isinstance(prior, dict) else prior[3]

    n = ss.get('n', ss.get('count', 0)) if isinstance(ss, dict) else ss[0]
    x_bar = ss.get('mean', ss.get('x_bar', 0.0)) if isinstance(ss, dict) else ss[1]
    s2 = ss.get('var', ss.get('s2', 0.0)) if isinstance(ss, dict) else ss[2]

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * n * s2 + 0.5 * kappa0 * n * (x_bar - mu0) ** 2 / kappa_n

    return {'mu0': mu_n, 'kappa0': kappa_n, 'alpha0': alpha_n, 'beta0': beta_n}
