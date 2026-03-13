from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""



import numpy as np
import icontract
from typing import Any
from ageoa.ghost.registry import register_atom
from .state_models import BPState


# Domain-specific type aliases
from .witnesses import (
    witness_initialize_message_passing_state,
    witness_run_loopy_message_passing_and_belief_query,
)


@register_atom(witness_initialize_message_passing_state)
@icontract.require(lambda pgm: pgm is not None, "pgm cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def initialize_message_passing_state(pgm: Any, state: BPState) -> tuple[object, BPState]:
    """Build the immutable loopy-belief-propagation state from the PGM.

    Args:
        pgm: Factor graph structure with variable/factor neighborhoods.
        state: BPState containing cross-window persistent state.

    Returns:
        Tuple of initialized state object and updated BPState.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_run_loopy_message_passing_and_belief_query)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda num_iter: isinstance(num_iter, int), "num_iter must be int")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def run_loopy_message_passing_and_belief_query(
    state_in: BPState, v_name: str, num_iter: int, state: BPState
) -> tuple[tuple[np.ndarray, object], BPState]:
    """Run loopy message-passing iterations and return queried variable belief.

    Args:
        state_in: Immutable input snapshot with msg/msg_new/t.
        v_name: Valid variable name in the PGM.
        num_iter: Number of message-passing iterations, >= 0.
        state: BPState containing cross-window persistent state.

    Returns:
        Tuple of (belief, state_out) and updated BPState.
    """
    raise NotImplementedError("Wire to original implementation")
