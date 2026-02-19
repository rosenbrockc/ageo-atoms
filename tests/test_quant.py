import pytest
from ageoa.quant_engine.atoms import calculate_ofi, execute_vwap, execute_pov, execute_passive
from ageoa.quant_engine.state_models import LimitQueueState

def test_quant_pipeline():
    state = LimitQueueState(
        strategy="vwap",
        risk_limit=1000.0,
        orders_ahead=1000,
        my_qty=500,
        is_filled=False,
        ofi_stream=[]
    )
    
    # 1. OFI Calc
    ofi, state = calculate_ofi(100.0, 100, 101.0, 100, 50, state)
    assert ofi == 0.0
    assert len(state.ofi_stream) == 1
    
    # 2. VWAP Execution
    _, state = execute_vwap(100, state)
    # participation 0.1 * 100 = 10. new qty = 500 - 10 = 490
    assert state.my_qty == 490
    
    # Reset for POV
    state = state.model_copy(update={"strategy": "pov", "orders_ahead": 100, "my_qty": 500})
    
    # 3. POV Execution (Queue ahead)
    _, state = execute_pov(50, state)
    assert state.orders_ahead == 50
    assert state.my_qty == 500
    
    # 4. POV Execution (Queue cleared)
    _, state = execute_pov(100, state)
    # 50 ahead - 100 trade = -50 remainder.
    # Logic: if orders_ahead > 0: ...
    # Wait, my logic in atom:
    # if orders_ahead > 0: orders_ahead -= trade; if orders_ahead < 0: remainder=abs; my_qty -= remainder; orders_ahead=0
    assert state.orders_ahead == 0
    assert state.my_qty == 450
