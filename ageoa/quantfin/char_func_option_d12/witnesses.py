from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal, ANYTHING
import networkx as nx  # type: ignore


def witness_charfuncoption(arg0: AbstractArray, cf: AbstractArray, charFuncMart: AbstractArray, d: AbstractArray, damp: AbstractArray, damp_prime: AbstractArray, disc: AbstractArray, exp: AbstractArray, f: AbstractArray, fg: AbstractArray, func1: AbstractArray, func2: AbstractArray, i: AbstractArray, intF: AbstractArray, k: AbstractArray, leftTerm: AbstractArray, log: AbstractArray, model: AbstractArray, opt: AbstractArray, p1: AbstractArray, p2: AbstractArray, pi: AbstractArray, q: AbstractArray, realPart: AbstractArray, rightTerm: AbstractArray, s: AbstractArray, strike: AbstractArray, tmat: AbstractArray, v: AbstractArray, v_prime: AbstractArray, x: AbstractArray, yc: AbstractArray) -> AbstractArray:
    """Ghost witness for Charfuncoption."""
    result = AbstractArray(
        shape=cf.shape,
        dtype="float64",
    )
    return result

def witness_f(exp: AbstractArray, i: AbstractArray, k: AbstractArray, leftTerm: AbstractArray, realPart: AbstractArray, rightTerm: AbstractArray, v: AbstractArray, v_prime: AbstractArray) -> AbstractArray:
    """Ghost witness for F."""
    result = AbstractArray(
        shape=exp.shape,
        dtype="float64",
    )
    return result

def witness_cf(charFuncMart: AbstractArray, fg: AbstractArray, model: AbstractArray, tmat: AbstractArray, x: AbstractArray) -> AbstractArray:
    """Ghost witness for Cf."""
    result = AbstractArray(
        shape=charFuncMart.shape,
        dtype="float64",
    )
    return result
