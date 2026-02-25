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

def witness_date(offset: AbstractArray) -> AbstractArray:
    """Ghost witness for Date."""
    result = AbstractArray(
        shape=offset.shape,
        dtype="float64",
    )
    return result

def witness_date(year: AbstractArray, dayinyear: AbstractArray) -> AbstractArray:
    """Ghost witness for Date."""
    result = AbstractArray(
        shape=year.shape,
        dtype="float64",
    )
    return result

def witness_show(io: AbstractArray, d: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_time(hour: AbstractArray, minute: AbstractArray, second: AbstractArray) -> AbstractArray:
    """Ghost witness for Time."""
    result = AbstractArray(
        shape=hour.shape,
        dtype="float64",
    )
    return result

def witness_time(secondinday: AbstractArray, fraction: AbstractArray) -> AbstractArray:
    """Ghost witness for Time."""
    result = AbstractArray(
        shape=secondinday.shape,
        dtype="float64",
    )
    return result

def witness_time(secondinday: AbstractArray) -> AbstractArray:
    """Ghost witness for Time."""
    result = AbstractArray(
        shape=secondinday.shape,
        dtype="float64",
    )
    return result

def witness_show(io: AbstractArray, t: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_datetime(year: AbstractArray, month: AbstractArray, day: AbstractArray, hour: AbstractArray, min: AbstractArray, sec: AbstractArray) -> AbstractArray:
    """Ghost witness for Datetime."""
    result = AbstractArray(
        shape=year.shape,
        dtype="float64",
    )
    return result

def witness_datetime(s: AbstractArray) -> AbstractArray:
    """Ghost witness for Datetime."""
    result = AbstractArray(
        shape=s.shape,
        dtype="float64",
    )
    return result

def witness_datetime(seconds: AbstractArray) -> AbstractArray:
    """Ghost witness for Datetime."""
    result = AbstractArray(
        shape=seconds.shape,
        dtype="float64",
    )
    return result
