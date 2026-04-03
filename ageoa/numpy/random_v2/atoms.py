"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import witness_continuousmultivariatesampler, witness_discreteeventsampler, witness_combinatoricssampler
@register_atom(witness_continuousmultivariatesampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda mean: mean is not None, "mean cannot be None")
@icontract.require(lambda cov: cov is not None, "cov cannot be None")
@icontract.require(lambda alpha: alpha is not None, "alpha cannot be None")
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "ContinuousMultivariateSampler all outputs must not be None")
def continuousmultivariatesampler(
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: np.ndarray,
    size: int | tuple[int, ...] | None = None,
    check_valid: str = "warn",
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw from NumPy's multivariate normal and dirichlet helpers using shared batch settings.

    Args:
        mean: Mean vector for the multivariate normal draw.
        cov: Covariance matrix for the multivariate normal draw.
        alpha: Concentration parameters for the Dirichlet draw.
        size: Optional batch size passed through to both NumPy draws.
        check_valid: Covariance validation mode for ``multivariate_normal``.
        tol: Covariance validation tolerance for ``multivariate_normal``.

    Returns:
        Tuple of multivariate-normal samples and Dirichlet samples.
    """
    mvn_samples = np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol)
    dirichlet_samples = np.random.dirichlet(alpha, size=size)
    return mvn_samples, dirichlet_samples

@register_atom(witness_discreteeventsampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda pvals: pvals is not None, "pvals cannot be None")
@icontract.ensure(lambda result: result is not None, "DiscreteEventSampler output must not be None")
def discreteeventsampler(
    n: int,
    pvals: np.ndarray,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Draw integer count vectors from NumPy's multinomial helper.

    Args:
        n: Total number of trials.
        pvals: Outcome probabilities for each bucket.
        size: Optional batch size for repeated draws.

    Returns:
        Multinomial count vector or batch of count vectors.
    """
    return np.random.multinomial(n, pvals, size=size)

@register_atom(witness_combinatoricssampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "CombinatoricsSampler all outputs must not be None")
def combinatoricssampler(
    x: int | np.ndarray,
    a: int | np.ndarray,
    size: int | tuple[int, ...] | None = None,
    replace: bool = True,
    p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly permute and sample from NumPy sequences.

    Args:
        x: Input for ``np.random.permutation``.
        a: Population input for ``np.random.choice``.
        size: Optional output shape for ``choice``.
        replace: Whether ``choice`` samples with replacement.
        p: Optional selection probabilities for ``choice``.

    Returns:
        Tuple of permuted input and sampled selection.
    """
    permuted = np.random.permutation(x)
    selected = np.random.choice(a, size=size, replace=replace, p=p)
    return permuted, selected
