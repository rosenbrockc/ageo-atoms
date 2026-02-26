"""Auto-generated stateful atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

# Import the original class for __new__ instantiation
# from <source_module> import QueueTracker

# State model should be imported from the generated state_models module
# from <state_module> import OrderState

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_initializeorderstate)
@icontract.require(lambda my_qty: isinstance(my_qty, (float, int, np.number)), "my_qty must be numeric")
@icontract.require(lambda orders_ahead: isinstance(orders_ahead, (float, int, np.number)), "orders_ahead must be numeric")
@icontract.ensure(lambda result: result is not None, "InitializeOrderState output must not be None")
def initializeorderstate(my_order_id: str, my_qty: float, orders_ahead: float, state: OrderState) -> tuple[OrderState, OrderState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Initializes the state of a new order within the queue, capturing its initial quantity and the volume of orders ahead of it.

    Args:
        my_order_id: Unique identifier for the order.
        my_qty: Initial quantity of the order.
        orders_ahead: Initial quantity of orders in the queue ahead of this one.
        state: OrderState object containing cross-window persistent state.

    Returns:
        tuple[A data structure representing the initial state, containing my_qty, orders_ahead, and is_filled status., OrderState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = QueueTracker.__new__(QueueTracker)
    obj.is_filled = state.is_filled
    obj.my_qty = state.my_qty
    obj.orders_ahead = state.orders_ahead
    new_state = state.model_copy(update={
        "is_filled": obj.is_filled,
        "my_qty": obj.my_qty,
        "orders_ahead": obj.orders_ahead,
    })
    result = obj.initial_order_state
    return result, new_state

@register_atom(witness_updatequeueontrade)
@icontract.require(lambda trade_qty: isinstance(trade_qty, (float, int, np.number)), "trade_qty must be numeric")
@icontract.ensure(lambda result: result is not None, "UpdateQueueOnTrade output must not be None")
def updatequeueontrade(current_order_state: OrderState, trade_qty: float, state: OrderState) -> tuple[OrderState, OrderState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Processes an incoming market trade, reducing the queue ahead and potentially filling the order. This is a deterministic state transition.

    Args:
        current_order_state: The current state of the order before the trade.
        trade_qty: The quantity of the executed market trade.
        state: OrderState object containing cross-window persistent state.

    Returns:
        tuple[A new data structure representing the updated state of the order after the trade., OrderState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = QueueTracker.__new__(QueueTracker)
    obj.is_filled = state.is_filled
    obj.my_qty = state.my_qty
    obj.orders_ahead = state.orders_ahead
    obj.process_trade(current_order_state, trade_qty)
    obj.fill_my_order(current_order_state, trade_qty)
    new_state = state.model_copy(update={
        "is_filled": obj.is_filled,
        "my_qty": obj.my_qty,
        "orders_ahead": obj.orders_ahead,
    })
    result = obj.next_order_state
    return result, new_state
