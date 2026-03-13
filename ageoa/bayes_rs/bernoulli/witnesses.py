from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_bernoulli_probabilistic_oracle(p: AbstractScalar, x: AbstractArray) -> AbstractArray:
    """Ghost witness for Bernoulli_Probabilistic_Oracle."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result
