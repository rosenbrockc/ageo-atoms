"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

# Witness functions used as atom registration keys
witness_continuousmultivariatesampler: None = None
witness_discreteeventsampler: None = None
witness_combinatoricssampler: None = None
@register_atom(witness_continuousmultivariatesampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda mean: isinstance(mean, (float, int, np.number)), "mean must be numeric")
@icontract.require(lambda cov: isinstance(cov, (float, int, np.number)), "cov must be numeric")
@icontract.require(lambda alpha: isinstance(alpha, (float, int, np.number)), "alpha must be numeric")
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "ContinuousMultivariateSampler all outputs must not be None")
def continuousmultivariatesampler(mean: np.ndarray, cov: np.ndarray, alpha: np.ndarray, size: int | tuple[int, ...] | None, check_valid: str, tol: float) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Draws samples from continuous multivariate distributions (Multivariate Normal and Dirichlet). These are the canonical prior/posterior distributions in Bayesian models — MVN for latent Gaussian states and Dirichlet for simplex-valued concentration parameters. Both are stateless, producing IID draws given distribution parameters.

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
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_discreteeventsampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda pvals: isinstance(pvals, (float, int, np.number)), "pvals must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "DiscreteEventSampler output must not be None")
def discreteeventsampler(n: int, pvals: np.ndarray, size: int | tuple[int, ...] | None) -> np.ndarray:  # type: ignore[type-arg]
    """Draws integer count vectors from the Multinomial distribution. Represents a single stateless stochastic step that maps a probability simplex (pvals) and a trial count (n) to an observed frequency vector. Commonly used to simulate categorical observations in generative models and particle filters.

    Args:
        n: total number of trials, n >= 0
        pvals: non-negative; must sum to <= 1 (last bucket absorbs remainder)
        size: determines output batch shape

    Returns:
        each row sums to n; counts[i] >= 0
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_combinatoricssampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda p: isinstance(p, (float, int, np.number)), "p must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "CombinatoricsSampler all outputs must not be None")
def combinatoricssampler(x: int | np.ndarray, axis: int, a: int | np.ndarray, size: int | tuple[int, ...] | None, replace: bool, p: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Performs combinatorial random operations on sequences or index spaces: random permutation of elements (permutation) and random selection with or without replacement (choice). Both operate purely on input arrays with no persistent state, making them reusable as shuffling or resampling kernels inside SMC or data-augmentation pipelines.

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
    raise NotImplementedError("Wire to original implementation")