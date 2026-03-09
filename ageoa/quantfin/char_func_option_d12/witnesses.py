"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_charfuncoption(: AbstractArray, cf: AbstractArray, charFuncMart: AbstractArray, d: AbstractArray, damp: AbstractArray, damp': AbstractArray, disc: AbstractArray, exp: AbstractArray, f: AbstractArray, fg: AbstractArray, func1: AbstractArray, func2: AbstractArray, i: AbstractArray, intF: AbstractArray, k: AbstractArray, leftTerm: AbstractArray, log: AbstractArray, model: AbstractArray, opt: AbstractArray, p1: AbstractArray, p2: AbstractArray, pi: AbstractArray, q: AbstractArray, realPart: AbstractArray, rightTerm: AbstractArray, s: AbstractArray, strike: AbstractArray, tmat: AbstractArray, v: AbstractArray, v': AbstractArray, x: AbstractArray, yc: AbstractArray) -> AbstractArray:
    """Ghost witness for Charfuncoption."""
    result = AbstractArray(
        shape=.shape,
        dtype="float64",
    )
    return result

def witness_f(exp: AbstractArray, i: AbstractArray, k: AbstractArray, leftTerm: AbstractArray, realPart: AbstractArray, rightTerm: AbstractArray, v: AbstractArray, v': AbstractArray) -> AbstractArray:
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
