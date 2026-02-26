"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
except ImportError:
    pass

def witness_initializeorderstate(event_shape: tuple[int, ...], family: str = "normal") -> AbstractDistribution:
    """Ghost witness for prior init: InitializeOrderState."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,
    )

def witness_updatequeueontrade(current_order_state: AbstractArray, trade_qty: AbstractArray, state: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for UpdateQueueOnTrade."""
    result = AbstractArray(
        shape=current_order_state.shape,
        dtype="float64",
    )
    return result, state
