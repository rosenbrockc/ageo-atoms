from __future__ import annotations
from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar

def witness_calculate_ofi(bid_px: float, bid_qty: int, ask_px: float, ask_qty: int, trade_qty: int, state: AbstractSignal) -> tuple[float, AbstractSignal]:
    """Ghost witness for Calculate OFI."""
    return 0.0, state

def witness_execute_vwap(trade_qty: int, state: AbstractSignal) -> tuple[None, AbstractSignal]:
    """Ghost witness for Execute VWAP."""
    return None, state

def witness_execute_pov(trade_qty: int, state: AbstractSignal) -> tuple[None, AbstractSignal]:
    """Ghost witness for Execute POV."""
    return None, state

def witness_execute_passive(trade_qty: int, state: AbstractSignal) -> tuple[None, AbstractSignal]:
    """Ghost witness for Execute Passive."""
    return None, state
