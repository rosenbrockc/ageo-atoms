from __future__ import annotations

from ageoa.institutional_quant_engine.queue_estimator.atoms import (
    initializeorderstate,
    updatequeueontrade,
)


def test_initializeorderstate_builds_order_state() -> None:
    state = initializeorderstate("order-1", 10.0)

    assert state.my_order_id == "order-1"
    assert state.my_qty == 10.0
    assert state.orders_ahead == 10000.0
    assert state.is_filled is False


def test_updatequeueontrade_consumes_queue_ahead_first() -> None:
    state = initializeorderstate("order-1", 10.0).model_copy(update={"orders_ahead": 25.0})

    next_state = updatequeueontrade(state, 12.0)

    assert next_state.orders_ahead == 13.0
    assert next_state.my_qty == 10.0
    assert next_state.is_filled is False


def test_updatequeueontrade_fills_order_after_queue_ahead_is_cleared() -> None:
    state = initializeorderstate("order-1", 10.0).model_copy(update={"orders_ahead": 5.0})

    next_state = updatequeueontrade(state, 12.0)

    assert next_state.orders_ahead == 0.0
    assert next_state.my_qty == 3.0
    assert next_state.is_filled is False


def test_updatequeueontrade_marks_state_filled_when_qty_is_exhausted() -> None:
    state = initializeorderstate("order-1", 4.0).model_copy(update={"orders_ahead": 1.0})

    next_state = updatequeueontrade(state, 7.0)

    assert next_state.orders_ahead == 0.0
    assert next_state.my_qty == 0.0
    assert next_state.is_filled is True
