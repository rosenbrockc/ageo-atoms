"""Stateless atoms for Institutional Quant Engine."""

from __future__ import annotations

from typing import Tuple

import icontract

from ageoa.ghost.registry import register_atom
from ageoa.quant_engine.state_models import LimitQueueState
from ageoa.quant_engine.witnesses import (
    witness_calculate_ofi,
    witness_execute_passive,
    witness_execute_pov,
    witness_execute_vwap,
)


def _non_negative(value: int) -> int:
    return max(0, int(value))


@register_atom(witness_calculate_ofi)
@icontract.require(lambda bid_qty, ask_qty: bid_qty >= 0 and ask_qty >= 0, "Quantities must be non-negative")
def calculate_ofi(
    bid_px: float,
    bid_qty: int,
    ask_px: float,
    ask_qty: int,
    trade_qty: int,
    state: LimitQueueState,
) -> Tuple[float, LimitQueueState]:
    """Calculate net directional pressure from competing bid/ask flow levels and append to state stream.
    """
    ofi = float(bid_qty - ask_qty) * 0.5

    current_stream = state.ofi_stream or []
    new_stream = current_stream + [ofi]

    new_state = state.model_copy(update={"ofi_stream": new_stream})
    return ofi, new_state


@register_atom(witness_execute_vwap)
@icontract.require(lambda trade_qty: trade_qty > 0, "Trade quantity must be positive")
@icontract.ensure(lambda result: (result[1].my_qty or 0) >= 0, "Inventory must remain non-negative")
def execute_vwap(trade_qty: int, state: LimitQueueState) -> Tuple[None, LimitQueueState]:
    """Volume-weighted average execution strategy logic.
    """
    participation_rate = 0.1
    current_qty = _non_negative(state.my_qty or 0)
    fill = min(current_qty, int(trade_qty * participation_rate))
    new_qty = current_qty - fill

    new_state = state.model_copy(update={"my_qty": new_qty})
    return None, new_state


@register_atom(witness_execute_pov)
@icontract.require(lambda trade_qty: trade_qty > 0, "Trade quantity must be positive")
@icontract.ensure(lambda result: (result[1].my_qty or 0) >= 0, "Inventory must remain non-negative")
@icontract.ensure(lambda result: (result[1].orders_ahead or 0) >= 0, "orders_ahead must remain non-negative")
def execute_pov(trade_qty: int, state: LimitQueueState) -> Tuple[None, LimitQueueState]:
    """Proportional participation execution strategy logic.
    """
    orders_ahead = _non_negative(state.orders_ahead or 0)
    my_qty = _non_negative(state.my_qty or 0)

    if orders_ahead > 0:
        filled_against_queue = max(0, trade_qty - orders_ahead)
        orders_ahead = max(0, orders_ahead - trade_qty)
        my_qty = max(0, my_qty - filled_against_queue)
    else:
        my_qty = max(0, my_qty - trade_qty)

    new_state = state.model_copy(update={
        "orders_ahead": orders_ahead,
        "my_qty": my_qty,
    })
    return None, new_state


@register_atom(witness_execute_passive)
@icontract.require(lambda trade_qty: trade_qty > 0, "Trade quantity must be positive")
@icontract.ensure(lambda result: (result[1].my_qty or 0) >= 0, "Inventory must remain non-negative")
@icontract.ensure(lambda result: (result[1].orders_ahead or 0) >= 0, "orders_ahead must remain non-negative")
def execute_passive(trade_qty: int, state: LimitQueueState) -> Tuple[None, LimitQueueState]:
    """Default queue-priority execution logic.
    """
    orders_ahead = _non_negative(state.orders_ahead or 0)
    my_qty = _non_negative(state.my_qty or 0)

    if orders_ahead > 0:
        orders_ahead = max(0, orders_ahead - trade_qty)
        queue_overflow = max(0, trade_qty - (state.orders_ahead or 0))
        my_qty = max(0, my_qty - queue_overflow)
    else:
        my_qty = max(0, my_qty - trade_qty)

    new_state = state.model_copy(update={
        "orders_ahead": orders_ahead,
        "my_qty": my_qty,
    })
    return None, new_state
