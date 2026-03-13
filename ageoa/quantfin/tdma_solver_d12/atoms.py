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
from .witnesses import witness_cotraversevec, witness_tdmasolver

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_tdmasolver)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda aL: aL is not None, "aL cannot be None")
@icontract.require(lambda ai: ai is not None, "ai cannot be None")
@icontract.require(lambda b: b is not None, "b cannot be None")
@icontract.require(lambda bL: bL is not None, "bL cannot be None")
@icontract.require(lambda bi: bi is not None, "bi cannot be None")
@icontract.require(lambda c: c is not None, "c cannot be None")
@icontract.require(lambda c_prime: c_prime is not None, "c_prime cannot be None")
@icontract.require(lambda cL: cL is not None, "cL cannot be None")
@icontract.require(lambda cf: cf is not None, "cf cannot be None")
@icontract.require(lambda ci: ci is not None, "ci cannot be None")
@icontract.require(lambda ci1: ci1 is not None, "ci1 cannot be None")
@icontract.require(lambda ci1_prime: ci1_prime is not None, "ci1_prime cannot be None")
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.require(lambda d_prime: d_prime is not None, "d_prime cannot be None")
@icontract.require(lambda dL: dL is not None, "dL cannot be None")
@icontract.require(lambda df: df is not None, "df cannot be None")
@icontract.require(lambda di: di is not None, "di cannot be None")
@icontract.require(lambda di1_prime: di1_prime is not None, "di1_prime cannot be None")
@icontract.require(lambda forM_: forM_ is not None, "forM_ cannot be None")
@icontract.require(lambda fromList: fromList is not None, "fromList cannot be None")
@icontract.require(lambda head: head is not None, "head cannot be None")
@icontract.require(lambda last: last is not None, "last cannot be None")
@icontract.require(lambda length: length is not None, "length cannot be None")
@icontract.require(lambda map: map is not None, "map cannot be None")
@icontract.require(lambda new: new is not None, "new cannot be None")
@icontract.require(lambda read: read is not None, "read cannot be None")
@icontract.require(lambda reverse: reverse is not None, "reverse cannot be None")
@icontract.require(lambda runST: runST is not None, "runST cannot be None")
@icontract.require(lambda thaw: thaw is not None, "thaw cannot be None")
@icontract.require(lambda toList: toList is not None, "toList cannot be None")
@icontract.require(lambda unsafeFreeze: unsafeFreeze is not None, "unsafeFreeze cannot be None")
@icontract.require(lambda write: write is not None, "write cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda xi1: xi1 is not None, "xi1 cannot be None")
@icontract.require(lambda xn: xn is not None, "xn cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Tdmasolver output must not be None")
def tdmasolver(a: Any, aL: Any, ai: Any, b: Any, bL: Any, bi: Any, c: Any, c_prime: Any, cL: Any, cf: Any, ci: Any, ci1: Any, ci1_prime: Any, d: Any, d_prime: Any, dL: Any, df: Any, di: Any, di1_prime: Any, forM_: Any, fromList: Any, head: Any, last: Any, length: Any, map: Any, new: Any, read: Any, reverse: Any, runST: Any, thaw: Any, toList: Any, unsafeFreeze: Any, write: Any, x: Any, xi1: Any, xn: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_cotraversevec)
@icontract.ensure(lambda result, **kwargs: result is not None, "Cotraversevec output must not be None")
def cotraversevec(enumFromN: Any, f: Any, fmap: Any, i: Any, l: Any, m: Any, map: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for haskell implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def tdmasolver_ffi(a, aL, ai, b, bL, bi, c, c_prime, cL, cf, ci, ci1, ci1_prime, d, d_prime, dL, df, di, di1_prime, forM_, fromList, head, last, length, map, new, read, reverse, runST, thaw, toList, unsafeFreeze, write, x, xi1, xn):
    """FFI bridge to Haskell implementation of Tdmasolver."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./tdmasolver.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(a, aL, ai, b, bL, bi, c, c_prime, cL, cf, ci, ci1, ci1_prime, d, d_prime, dL, df, di, di1_prime, forM_, fromList, head, last, length, map, new, read, reverse, runST, thaw, toList, unsafeFreeze, write, x, xi1, xn)

def cotraversevec_ffi(enumFromN, f, fmap, i, l, m, map):
    """FFI bridge to Haskell implementation of Cotraversevec."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./cotraversevec.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(enumFromN, f, fmap, i, l, m, map)
