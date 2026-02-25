"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
except ImportError:
    pass

_MEMO_CACHE: dict = {}


def _clear_memo_cache() -> None:
    """Reset the memoization cache between iterations."""
    _MEMO_CACHE.clear()


def witness_initialize_message_passing_state(event_shape: tuple[int, ...], family: str = "normal") -> AbstractDistribution:
    """Ghost witness for prior init: initialize_message_passing_state."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,
    )

def witness_run_loopy_message_passing_and_belief_query(state_in: AbstractArray, v_name: AbstractArray, num_iter: AbstractArray) -> AbstractArray:
    """Ghost witness for message-passing: run_loopy_message_passing_and_belief_query."""
    _cache_key = ("run_loopy_message_passing_and_belief_query",)
    if _cache_key in _MEMO_CACHE:
        return _MEMO_CACHE[_cache_key]
    result = AbstractArray(shape=state_in.shape, dtype=state_in.dtype)
    _MEMO_CACHE[_cache_key] = result
    return result
