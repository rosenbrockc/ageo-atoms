"""Stateless atoms for Institutional Quant Engine."""

from __future__ import annotations
import numpy as np
import icontract
from typing import Tuple, Any, List
from ageoa.ghost.registry import register_atom
from ageoa.quant_engine.state_models import LimitQueueState
from ageoa.quant_engine.witnesses import (
    witness_calculate_ofi,
    witness_execute_vwap,
    witness_execute_pov,
    witness_execute_passive
)

@register_atom(witness_calculate_ofi)
@icontract.require(lambda bid_qty, ask_qty: bid_qty >= 0 and ask_qty >= 0, "Quantities must be non-negative")
def calculate_ofi(bid_px: float, bid_qty: int, ask_px: float, ask_qty: int, trade_qty: int, state: LimitQueueState) -> Tuple[float, LimitQueueState]:
    """Calculates Order Flow Imbalance."""
    # Functional core logic
    ofi = float(bid_qty - ask_qty) * 0.5
    
    # State update
    current_stream = state.ofi_stream or []
    new_stream = current_stream + [ofi]
    
    new_state = state.model_copy(update={
        "ofi_stream": new_stream
    })
    
    return ofi, new_state

@register_atom(witness_execute_vwap)
@icontract.require(lambda trade_qty: trade_qty > 0, "Trade quantity must be positive")
def execute_vwap(trade_qty: int, state: LimitQueueState) -> Tuple[None, LimitQueueState]:
    """VWAP execution strategy logic."""
    participation_rate = 0.1
    fill = int(trade_qty * participation_rate)
    new_qty = (state.my_qty or 0) - fill
    
    new_state = state.model_copy(update={
        "my_qty": new_qty
    })
    return None, new_state

@register_atom(witness_execute_pov)
@icontract.require(lambda trade_qty: trade_qty > 0, "Trade quantity must be positive")
def execute_pov(trade_qty: int, state: LimitQueueState) -> Tuple[None, LimitQueueState]:
    """POV execution strategy logic."""
    orders_ahead = state.orders_ahead or 0
    my_qty = state.my_qty or 0
    
    if orders_ahead > 0:
        orders_ahead -= trade_qty
        if orders_ahead < 0:
            # Remainder fills my order
            remainder = abs(orders_ahead)
            my_qty -= remainder
            orders_ahead = 0
    else:
        my_qty -= trade_qty
        
    new_state = state.model_copy(update={
        "orders_ahead": orders_ahead,
        "my_qty": my_qty
    })
    return None, new_state

@register_atom(witness_execute_passive)
@icontract.require(lambda trade_qty: trade_qty > 0, "Trade quantity must be positive")
def execute_passive(trade_qty: int, state: LimitQueueState) -> Tuple[None, LimitQueueState]:
    """Default passive execution logic."""
    orders_ahead = state.orders_ahead or 0
    my_qty = state.my_qty or 0
    
    if orders_ahead > 0:
        orders_ahead -= trade_qty
        if orders_ahead < 0:
            remainder = abs(orders_ahead)
            my_qty -= min(my_qty, remainder)
            orders_ahead = 0
    else:
        my_qty -= min(my_qty, trade_qty)
        
    new_state = state.model_copy(update={
        "orders_ahead": orders_ahead,
        "my_qty": my_qty
    })
    return None, new_state
