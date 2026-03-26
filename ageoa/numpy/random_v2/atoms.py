"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import witness_continuousmultivariatesampler, witness_discreteeventsampler, witness_combinatoricssampler
@register_atom(witness_continuousmultivariatesampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda mean: isinstance(mean, (float, int, np.number)), "mean must be numeric")
@icontract.require(lambda cov: isinstance(cov, (float, int, np.number)), "cov must be numeric")
@icontract.require(lambda alpha: isinstance(alpha, (float, int, np.number)), "alpha must be numeric")
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "ContinuousMultivariateSampler all outputs must not be None")
def continuousmultivariatesampler(mean: np.ndarray, cov: np.ndarray, alpha: np.ndarray, size: int | tuple[int, ...] | None, check_valid: str, tol: float) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Draws samples from continuous multivariate distributions: Multivariate Normal (MVN) and Dirichlet. Produces independent, identically distributed (IID) draws given distribution parameters.

    Args:
        mean: length must match leading dim of cov
        cov: must be symmetric positive-semidefinite
        alpha: all elements > 0
        size: determines output batch shape
        check_valid: controls covariance validation
        tol: tolerance for covariance symmetry check

    Returns:
        mvn_samples: drawn IID from N(mean, cov)
        dirichlet_samples: rows sum to 1; drawn from Dir(alpha)
    """
    mvn_samples = np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol)
    dirichlet_samples = np.random.dirichlet(alpha, size=size)
    return mvn_samples, dirichlet_samples

@register_atom(witness_discreteeventsampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda pvals: isinstance(pvals, (float, int, np.number)), "pvals must be numeric")
@icontract.ensure(lambda result: result is not None, "DiscreteEventSampler output must not be None")
def discreteeventsampler(n: int, pvals: np.ndarray, size: int | tuple[int, ...] | None) -> np.ndarray:  # type: ignore[type-arg]
    """Draws integer count vectors from the Multinomial distribution. Represents a single stateless stochastic step that maps a probability simplex (pvals) and a trial count (n) to an observed frequency vector. Commonly used to simulate categorical observations in generative models and particle filters.

    Args:
        n: total number of trials, n >= 0
        pvals: non-negative; must sum to <= 1 (last bucket absorbs remainder)
        size: determines output batch shape

    Returns:
        each row sums to n; counts[i] >= 0
    """
    return np.random.multinomial(n, pvals, size=size)

@register_atom(witness_combinatoricssampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda p: isinstance(p, (float, int, np.number)), "p must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "CombinatoricsSampler all outputs must not be None")
def combinatoricssampler(x: int | np.ndarray, axis: int, a: int | np.ndarray, size: int | tuple[int, ...] | None, replace: bool, p: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Randomly permutes or selects elements from a sequence. Permutation shuffles all elements; choice draws a sample with or without replacement.

    Args:
        x: for permutation; int must be >= 0
        axis: axis along which to permute / select; 0-based
        a: for choice; int must be >= 1
        size: output shape for choice
        replace: sampling with (True) or without (False) replacement
        p: selection probabilities; must sum to 1 if provided

    Returns:
        permuted: all elements of x present exactly once
        selected: drawn from a along axis; respects replace flag
    """
    permuted = np.random.permutation(x)
    selected = np.random.choice(a, size=size, replace=replace, p=p)
    return permuted, selected