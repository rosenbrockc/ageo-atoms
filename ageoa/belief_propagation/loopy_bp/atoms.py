"""Auto-generated stateful atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

# Import the original class for __new__ instantiation
# from <source_module> import loopy_belief_propagation

# State model should be imported from the generated state_models module
# from <state_module> import BPState

# Witness functions should be imported from the generated witnesses module
from typing import Any, TypeVar
witness_initialize_message_passing_state: Any = None
witness_run_loopy_message_passing_and_belief_query: Any = None

ProbabilisticGraphModel = Any
BPState = Any
BPStateModelSpec = Any
NormalizedVector = Any
pgm = TypeVar("pgm")
msg = TypeVar("msg")
msg_new = TypeVar("msg_new")
t = TypeVar("t")
loopy_belief_propagation: Any = object
@register_atom(witness_initialize_message_passing_state)
@icontract.require(lambda pgm: pgm is not None, "pgm cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "initialize_message_passing_state output must not be None")
def initialize_message_passing_state(pgm: ProbabilisticGraphModel, state: BPState) -> tuple[BPStateModelSpec[pgm, msg, msg_new, t], BPState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Builds the immutable loopy-belief-propagation state object from the PGM, including message tables/buffers and iteration counter.

    Args:
        pgm: Factor graph structure with variable/factor neighborhoods and factor tables.
        state: BPState object containing cross-window persistent state.

    Returns:
        tuple[New object; includes initialized factor->variable and variable->factor message storage plus t=0., BPState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = loopy_belief_propagation.__new__(loopy_belief_propagation)
@register_atom(witness_run_loopy_message_passing_and_belief_query)  # type: ignore[untyped-decorator]
    obj.msg_new = state.msg_new
    obj.t = state.t
    new_state = state.model_copy(update={
        "pgm": obj.pgm,
        "msg": obj.msg,
        "msg_new": obj.msg_new,
        "t": obj.t,
    })
    result = obj.state_out
    return result, new_state

@register_atom(witness_run_loopy_message_passing_and_belief_query)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda v_name: v_name is not None, "v_name cannot be None")
@icontract.require(lambda num_iter: num_iter is not None, "num_iter cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "run_loopy_message_passing_and_belief_query all outputs must not be None")
def run_loopy_message_passing_and_belief_query(state_in: BPStateModelSpec[pgm, msg, msg_new, t], v_name: str, num_iter: int, state: BPState) -> tuple[tuple[NormalizedVector, BPStateModelSpec[pgm, msg, msg_new, t]], BPState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Runs pure loopy message-passing iterations (variable->factor and factor->variable updates with normalization) and returns the queried variable belief with an updated immutable state.

    Args:
        state_in: Immutable input snapshot; msg/msg_new/t are read and transformed into a new state.
        v_name: Must be a valid variable name in pgm.
        num_iter: num_iter >= 0.
        state: BPState object containing cross-window persistent state.

    Returns:
        tuple[tuple[belief, state_out], BPState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = loopy_belief_propagation.__new__(loopy_belief_propagation)
    obj.pgm = state.pgm
    obj.msg = state.msg
    obj.msg_new = state.msg_new
    obj.t = state.t
    obj.belief(state_in, v_name, num_iter)
    obj.get_variable2factor_msg(state_in, v_name, num_iter)
    obj.__compute_variable2factor_msg(state_in, v_name, num_iter)
    obj.get_factor2variable_msg(state_in, v_name, num_iter)
    obj.__compute_factor2variable_msg(state_in, v_name, num_iter)
    obj.__loop(state_in, v_name, num_iter)
    obj.__normalize_msg(state_in, v_name, num_iter)
    new_state = state.model_copy(update={
        "pgm": obj.pgm,
        "msg": obj.msg,
        "msg_new": obj.msg_new,
        "t": obj.t,
    })
    result = (obj.belief, obj.state_out)
    return result, new_state