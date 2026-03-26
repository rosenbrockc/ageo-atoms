from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_var(s: AbstractArray, t: AbstractArray, t_prime: AbstractArray, v: AbstractArray, vs: AbstractArray) -> AbstractArray:
    """Shape-and-type check for var. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=s.shape,
        dtype="float64",
    )
    return result

def witness_localvol(dwdt: AbstractArray, k: AbstractArray, otherwise: AbstractArray, rcurve: AbstractArray, s0: AbstractArray, solution: AbstractArray, sqrt: AbstractArray, t: AbstractArray, v: AbstractArray, w: AbstractArray) -> AbstractArray:
    """Shape-and-type check for localvol. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=dwdt.shape,
        dtype="float64",
    )
    return result

def witness_vol(x: AbstractArray) -> AbstractArray:
    """Shape-and-type check for vol. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_vol(interpolatedVs: AbstractArray, mats: AbstractArray, mats_prime: AbstractArray, quotes: AbstractArray, strike: AbstractArray, sts: AbstractArray, t: AbstractArray, tInterp: AbstractArray, timeFromZero: AbstractArray, vInterp: AbstractArray) -> AbstractArray:
    """Shape-and-type check for vol. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=interpolatedVs.shape,
        dtype="float64",
    )
    return result

def witness_allfort(map: AbstractArray, quotes: AbstractArray, sts: AbstractArray, t_prime: AbstractArray, x: AbstractArray) -> AbstractArray:
    """Shape-and-type check for allfort. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=map.shape,
        dtype="float64",
    )
    return result
