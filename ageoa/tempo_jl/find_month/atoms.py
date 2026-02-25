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

@register_atom(witness_date)
@icontract.require(lambda offset: offset is not None, "offset cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Date output must not be None")
def date(offset: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_date)
@icontract.require(lambda year: year is not None, "year cannot be None")
@icontract.require(lambda dayinyear: dayinyear is not None, "dayinyear cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Date output must not be None")
def date(year: Any, dayinyear: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, d: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_time)
@icontract.require(lambda hour: hour is not None, "hour cannot be None")
@icontract.require(lambda minute: minute is not None, "minute cannot be None")
@icontract.require(lambda second: second is not None, "second cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Time output must not be None")
def time(hour: Any, minute: Any, second: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_time)
@icontract.require(lambda secondinday: secondinday is not None, "secondinday cannot be None")
@icontract.require(lambda fraction: fraction is not None, "fraction cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Time output must not be None")
def time(secondinday: Any, fraction: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_time)
@icontract.require(lambda secondinday: secondinday is not None, "secondinday cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Time output must not be None")
def time(secondinday: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, t: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_datetime)
@icontract.ensure(lambda result, **kwargs: result is not None, "Datetime output must not be None")
def datetime(year: Any, month: Any, day: Any, hour: Any, min: Any, sec: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_datetime)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Datetime output must not be None")
def datetime(s: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_datetime)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Datetime output must not be None")
def datetime(seconds: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def date_ffi(offset):
    """FFI bridge to Julia implementation of Date."""
    return jl.eval("date(offset)")

def date_ffi(year, dayinyear):
    """FFI bridge to Julia implementation of Date."""
    return jl.eval("date(year, dayinyear)")

def show_ffi(io, d):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, d)")

def time_ffi(hour, minute, second):
    """FFI bridge to Julia implementation of Time."""
    return jl.eval("time(hour, minute, second)")

def time_ffi(secondinday, fraction):
    """FFI bridge to Julia implementation of Time."""
    return jl.eval("time(secondinday, fraction)")

def time_ffi(secondinday):
    """FFI bridge to Julia implementation of Time."""
    return jl.eval("time(secondinday)")

def show_ffi(io, t):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, t)")

def datetime_ffi(year, month, day, hour, min, sec):
    """FFI bridge to Julia implementation of Datetime."""
    return jl.eval("datetime(year, month, day, hour, min, sec)")

def datetime_ffi(s):
    """FFI bridge to Julia implementation of Datetime."""
    return jl.eval("datetime(s)")

def datetime_ffi(seconds):
    """FFI bridge to Julia implementation of Datetime."""
    return jl.eval("datetime(seconds)")
