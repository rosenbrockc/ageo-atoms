from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_execute_pov(trade_qty: AbstractScalar, state: AbstractSignal) -> tuple[None, AbstractSignal]:
    """Ghost witness for Proportional Participation Execution."""
    return None, state


def witness_execute_passive(trade_qty: AbstractScalar, state: AbstractSignal) -> tuple[None, AbstractSignal]:
    """Ghost witness for Queue-Priority Execution."""
    return None, state


def witness_calculate_ofi(bid_px: AbstractScalar, bid_qty: AbstractScalar, ask_px: AbstractScalar, ask_qty: AbstractScalar, trade_qty: AbstractScalar, state: AbstractSignal) -> tuple[AbstractScalar, AbstractSignal]:
    """Ghost witness for Order Flow Imbalance calculation."""
    ofi = AbstractScalar(dtype="float64")
    return ofi, state


def witness_execute_vwap(trade_qty: AbstractScalar, state: AbstractSignal) -> tuple[None, AbstractSignal]:
    """Ghost witness for VWAP execution."""
    return None, state
