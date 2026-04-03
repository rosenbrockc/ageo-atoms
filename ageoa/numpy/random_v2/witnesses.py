from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar


def witness_continuousmultivariatesampler(
    mean: AbstractArray,
    cov: AbstractArray,
    alpha: AbstractArray,
    size: AbstractScalar | AbstractArray | None,
    check_valid: AbstractScalar,
    tol: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray]:
    """Return shape metadata for the paired multivariate-normal and Dirichlet draws."""
    mvn = AbstractArray(shape=mean.shape, dtype="float64")
    dirichlet = AbstractArray(shape=alpha.shape, dtype="float64")
    return mvn, dirichlet

def witness_discreteeventsampler(n: AbstractScalar, pvals: AbstractArray, size: AbstractScalar | AbstractArray | None) -> AbstractArray:
    """Return count-vector metadata for the multinomial wrapper."""
    return AbstractArray(shape=pvals.shape, dtype="int64")

def witness_combinatoricssampler(
    x: AbstractArray | AbstractScalar,
    a: AbstractArray | AbstractScalar,
    size: AbstractScalar | AbstractArray | None,
    replace: AbstractScalar,
    p: AbstractArray | None,
) -> tuple[AbstractArray, AbstractArray]:
    """Return permutation and sampling metadata for the combinatorics wrapper."""
    source = x if isinstance(x, AbstractArray) else AbstractArray(shape=("N",), dtype="int64")
    choice_source = a if isinstance(a, AbstractArray) else AbstractArray(shape=("N",), dtype="int64")
    return (
        AbstractArray(shape=source.shape, dtype=source.dtype),
        AbstractArray(shape=choice_source.shape, dtype=choice_source.dtype),
    )
