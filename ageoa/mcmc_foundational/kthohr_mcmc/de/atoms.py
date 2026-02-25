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

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom("build_de_transition_kernel")  # type: ignore[untyped-decorator]
@icontract.require(lambda target_log_kernel: target_log_kernel is not None, "target_log_kernel cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "build_de_transition_kernel output must not be None")
def build_de_transition_kernel(target_log_kernel: object) -> object:
    """Creates a pure Differential Evolution transition kernel from the provided target log-density oracle.

    Args:
        target_log_kernel: Stateless log-density oracle; no persistent state mutation.

    Returns:
        Pure transition function; any stochastic state (e.g., RNG/PRNGKey) must be explicit input/output.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path

def build_de_transition_kernel_ffi(target_log_kernel: object) -> object:
    """FFI bridge to C++ implementation of build_de_transition_kernel."""
    _func_name = 'build_de_transition_kernel'
    _func_name = 'build_de_transition_kernel'
    _func = ctypes.CDLL(None)[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target_log_kernel)