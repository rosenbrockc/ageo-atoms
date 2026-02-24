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

@register_atom(witness_temper)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda r: r is not None, "r cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Temper output must not be None")
def temper(lf: Any, r: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda l: l is not None, "l cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, l: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda l: l is not None, "l cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, l: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_jitter)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Jitter output must not be None")
def jitter(rng: Any, lf: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_jitter)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Jitter output must not be None")
def jitter(rng: Any, lf: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda l: l is not None, "l cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, l: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_temper)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda r: r is not None, "r cannot be None")
@icontract.require(lambda step: step is not None, "step cannot be None")
@icontract.require(lambda n_steps: n_steps is not None, "n_steps cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Temper output must not be None")
def temper(lf: Any, r: Any, step: Any, n_steps: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_step)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Step output must not be None")
def step(lf: Any, h: Any, z: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def temper_ffi(lf, r):
    """FFI bridge to Julia implementation of Temper."""
    return jl.eval("temper(lf, r)")

def show_ffi(io, l):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, l)")

def show_ffi(io, l):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, l)")

def jitter_ffi(rng, lf):
    """FFI bridge to Julia implementation of Jitter."""
    return jl.eval("jitter(rng, lf)")

def jitter_ffi(rng, lf):
    """FFI bridge to Julia implementation of Jitter."""
    return jl.eval("jitter(rng, lf)")

def show_ffi(io, l):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, l)")

def temper_ffi(lf, r, step, n_steps):
    """FFI bridge to Julia implementation of Temper."""
    return jl.eval("temper(lf, r, step, n_steps)")

def step_ffi(lf, h, z):
    """FFI bridge to Julia implementation of Step."""
    return jl.eval("step(lf, h, z)")
