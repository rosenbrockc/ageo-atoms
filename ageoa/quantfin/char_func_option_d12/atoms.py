from __future__ import annotations
from typing import Any
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_cf, witness_charfuncoption, witness_f

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_charfuncoption)
@icontract.require(lambda cf: cf is not None, "cf cannot be None")
@icontract.require(lambda charFuncMart: charFuncMart is not None, "charFuncMart cannot be None")
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.require(lambda damp: damp is not None, "damp cannot be None")
@icontract.require(lambda damp_prime: damp_prime is not None, "damp_prime cannot be None")
@icontract.require(lambda disc: disc is not None, "disc cannot be None")
@icontract.require(lambda exp: exp is not None, "exp cannot be None")
@icontract.require(lambda f: f is not None, "f cannot be None")
@icontract.require(lambda fg: fg is not None, "fg cannot be None")
@icontract.require(lambda func1: func1 is not None, "func1 cannot be None")
@icontract.require(lambda func2: func2 is not None, "func2 cannot be None")
@icontract.require(lambda i: i is not None, "i cannot be None")
@icontract.require(lambda intF: intF is not None, "intF cannot be None")
@icontract.require(lambda k: k is not None, "k cannot be None")
@icontract.require(lambda leftTerm: leftTerm is not None, "leftTerm cannot be None")
@icontract.require(lambda log: log is not None, "log cannot be None")
@icontract.require(lambda model: model is not None, "model cannot be None")
@icontract.require(lambda opt: opt is not None, "opt cannot be None")
@icontract.require(lambda p1: p1 is not None, "p1 cannot be None")
@icontract.require(lambda p2: p2 is not None, "p2 cannot be None")
@icontract.require(lambda pi: pi is not None, "pi cannot be None")
@icontract.require(lambda q: q is not None, "q cannot be None")
@icontract.require(lambda realPart: realPart is not None, "realPart cannot be None")
@icontract.require(lambda rightTerm: rightTerm is not None, "rightTerm cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda strike: strike is not None, "strike cannot be None")
@icontract.require(lambda tmat: tmat is not None, "tmat cannot be None")
@icontract.require(lambda v: v is not None, "v cannot be None")
@icontract.require(lambda v_prime: v_prime is not None, "v_prime cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda yc: yc is not None, "yc cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Charfuncoption output must not be None")
def charfuncoption(arg0: Any, cf: Any, charFuncMart: Any, d: Any, damp: Any, damp_prime: Any, disc: Any, exp: Any, f: Any, fg: Any, func1: Any, func2: Any, i: Any, intF: Any, k: Any, leftTerm: Any, log: Any, model: Any, opt: Any, p1: Any, p2: Any, pi: Any, q: Any, realPart: Any, rightTerm: Any, s: Any, strike: Any, tmat: Any, v: Any, v_prime: Any, x: Any, yc: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_f)
@icontract.ensure(lambda result, **kwargs: result is not None, "F output must not be None")
def f(exp: Any, i: Any, k: Any, leftTerm: Any, realPart: Any, rightTerm: Any, v: Any, v_prime: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_cf)
@icontract.require(lambda charFuncMart: charFuncMart is not None, "charFuncMart cannot be None")
@icontract.require(lambda fg: fg is not None, "fg cannot be None")
@icontract.require(lambda model: model is not None, "model cannot be None")
@icontract.require(lambda tmat: tmat is not None, "tmat cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Cf output must not be None")
def cf(charFuncMart: Any, fg: Any, model: Any, tmat: Any, x: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for haskell implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def charfuncoption_ffi(arg0, cf, charFuncMart, d, damp, damp_prime, disc, exp, f, fg, func1, func2, i, intF, k, leftTerm, log, model, opt, p1, p2, pi, q, realPart, rightTerm, s, strike, tmat, v, v_prime, x, yc):
    """FFI bridge to Haskell implementation of Charfuncoption."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./charfuncoption.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(arg0, cf, charFuncMart, d, damp, damp_prime, disc, exp, f, fg, func1, func2, i, intF, k, leftTerm, log, model, opt, p1, p2, pi, q, realPart, rightTerm, s, strike, tmat, v, v_prime, x, yc)

def f_ffi(exp, i, k, leftTerm, realPart, rightTerm, v, v_prime):
    """FFI bridge to Haskell implementation of F."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./f.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(exp, i, k, leftTerm, realPart, rightTerm, v, v_prime)

def cf_ffi(charFuncMart, fg, model, tmat, x):
    """FFI bridge to Haskell implementation of Cf."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./cf.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(charFuncMart, fg, model, tmat, x)
