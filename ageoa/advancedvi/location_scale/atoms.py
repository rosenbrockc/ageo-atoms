from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_evaluate_log_probability_density

# juliacall unavailable; reimplemented in pure numpy


@register_atom(witness_evaluate_log_probability_density)
@icontract.require(lambda q: q is not None, "q cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result: result is not None, "evaluate_log_probability_density output must not be None")
def evaluate_log_probability_density(q: np.ndarray, z: np.ndarray) -> float:
    """Computes the log-probability density function (logpdf) for given inputs 'q' and 'z'. This is a stateless operation, likely corresponding to a specific probability distribution.

    Args:
        q: Input parameter for the logpdf calculation.
        z: Input parameter for the logpdf calculation.

    Returns:
        The resulting log-probability density.
    """
    # Location-scale Gaussian logpdf: q = [mu, log_sigma], z = sample
    d = len(q) // 2
    mu = q[:d]
    log_sigma = q[d:]
    sigma = np.exp(log_sigma)
    # log N(z; mu, sigma^2) = -0.5*d*log(2*pi) - sum(log_sigma) - 0.5*sum(((z-mu)/sigma)^2)
    return float(-0.5 * d * np.log(2 * np.pi) - np.sum(log_sigma) - 0.5 * np.sum(((z - mu) / sigma) ** 2))
