"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_posterior_randmodel)
@icontract.require(lambda pri: pri is not None, "pri cannot be None")
@icontract.require(lambda G: G is not None, "G cannot be None")
@icontract.require(lambda data: data is not None, "data cannot be None")
@icontract.ensure(lambda result: result is not None, "Posterior Randmodel output must not be None")
def posterior_randmodel(pri: Any, G: Any, data: Any) -> Any:
    """Posterior randmodel.

    Args:
        pri (Any): Description.
        G (Any): Description.
        data (Any): Description.

    Returns:
        Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_posterior_randmodel)
@icontract.require(lambda pri: pri is not None, "pri cannot be None")
@icontract.require(lambda G: G is not None, "G cannot be None")
@icontract.require(lambda data: data is not None, "data cannot be None")
@icontract.require(lambda w: w is not None, "w cannot be None")
@icontract.ensure(lambda result: result is not None, "Posterior Randmodel output must not be None")
def posterior_randmodel(pri: Any, G: Any, data: Any, w: Any) -> Any:
    """Posterior randmodel.

    Args:
        pri (Any): Description.
        G (Any): Description.
        data (Any): Description.
        w (Any): Description.

    Returns:
        Description.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def _posterior_randmodel_ffi(pri, G, data):
    """FFI bridge to Julia implementation of Posterior Randmodel."""
    return jl.eval("posterior_randmodel(pri, G, data)")

def _posterior_randmodel_ffi(pri, G, data, w):
    """FFI bridge to Julia implementation of Posterior Randmodel."""
    return jl.eval("posterior_randmodel(pri, G, data, w)")
