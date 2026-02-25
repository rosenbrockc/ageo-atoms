"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module
@register_atom(witness_bernoulli_probabilistic_oracle)  # type: ignore[untyped-decorator,name-defined]
@register_atom(witness_bernoulli_probabilistic_oracle)
@icontract.require(lambda p: isinstance(p, (float, int, np.number)), "p must be numeric")
def bernoulli_probabilistic_oracle(p: float, x: int | np.ndarray[Any, Any]) -> float | np.ndarray[Any, Any]:
    """Defines and evaluates a Bernoulli distribution as a pure probabilistic oracle with no persistent mutable state.

    Args:
        p: Probability parameter, 0 <= p <= 1
        x: Observed outcome(s), each value in {0,1} when likelihood/log_prob is evaluated

    Returns:
        Elementwise Bernoulli log-likelihood values
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

from __future__ import annotations

import ctypes.util
from pathlib import Path

def bernoulli_probabilistic_oracle_ffi(p: ctypes.c_void_p, x: ctypes.c_void_p) -> ctypes.c_void_p:
def bernoulli_probabilistic_oracle_ffi(p, x):
    """FFI bridge to Rust implementation of Bernoulli_Probabilistic_Oracle."""
def bernoulli_probabilistic_oracle_ffi(p: ctypes.c_void_p, x: ctypes.c_void_p) -> ctypes.c_void_p:
    _func_name = atom.method_names[0] if atom.method_names else 'bernoulli_probabilistic_oracle'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(p, x)