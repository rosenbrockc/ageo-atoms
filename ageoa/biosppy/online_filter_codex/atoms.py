from __future__ import annotations
"""Auto-generated stateful atom wrappers following the ageoa pattern."""

from typing import Any

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_filterstateinit, witness_filterstep

# Import the original class for __new__ instantiation
from biosppy.signals.tools import OnlineFilter

# State model should be imported from the generated state_models module
from .state_models import FilterParamState

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_filterstateinit)
@icontract.require(lambda b: b is not None, "b cannot be None")
@icontract.require(lambda a: a is not None, "a cannot be None")
def filterstateinit(b: Any, a: Any, state: FilterParamState) -> tuple[tuple[Any, Any, Any], FilterParamState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Initializes the IIR/FIR filter by storing numerator (b) and denominator (a) coefficients and bootstrapping the delay-line state vector (zi) to zeros. reset() is an integral part of this initialization stage and can also be re-invoked to flush the delay line between signals.

    Args:
        b: len(b) >= 1
        a: len(a) >= 1; a[0] != 0
        state: FilterParamState object containing cross-window persistent state.

    Returns:
        tuple[tuple[b, a, zi], FilterParamState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = OnlineFilter.__new__(OnlineFilter)
    obj.b = state.b
    obj.a = state.a
    obj.reset(b, a)
    new_state = state.model_copy(update={
        "b": obj.b,
        "a": obj.a,
    })
    result = (obj.b, obj.a, obj.zi)
    return result, new_state

@register_atom(witness_filterstep)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda b: b is not None, "b cannot be None")
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda zi: zi is not None, "zi cannot be None")
def filterstep(signal: Any, b: Any, a: Any, zi: Any, state: FilterParamState) -> tuple[tuple[Any, Any], FilterParamState]:
    """
    Applies the online IIR/FIR filter to an incoming signal chunk using the current delay-line state. Consumes zi as immutable state-in and produces a new zi as state-out, enabling stateless composition across successive calls for true online (sample-by-sample or block) filtering.

    Args:
        signal: dtype float; length >= 1
        b: from FilterStateInit
        a: from FilterStateInit
        zi: immutable; produced by FilterStateInit or previous FilterStep
        state: FilterParamState object containing cross-window persistent state.

    Returns:
        tuple[tuple[filtered_signal, zi_out], FilterParamState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = OnlineFilter.__new__(OnlineFilter)
    obj.b = state.b
    obj.a = state.a
    obj.filter(signal, b, a, zi)
    new_state = state.model_copy(update={
        "b": obj.b,
        "a": obj.a,
    })
    result = (obj.filtered_signal, obj.zi_out)
    return result, new_state
