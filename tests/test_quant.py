from ageoa.quant_engine.atoms import calculate_ofi, execute_passive, execute_pov, execute_vwap
from ageoa.quant_engine.state_models import LimitQueueState


class TestQuantPipeline:
    def test_ofi_and_vwap(self):
        state = LimitQueueState(
            strategy="vwap",
            risk_limit=1000.0,
            orders_ahead=1000,
            my_qty=500,
            is_filled=False,
            ofi_stream=[],
        )

        # 1. OFI Calc
        ofi, state = calculate_ofi(100.0, 100, 101.0, 100, 50, state)
        assert ofi == 0.0
        assert len(state.ofi_stream) == 1

        # 2. VWAP Execution
        _, state = execute_vwap(100, state)
        # participation 0.1 * 100 = 10. new qty = 500 - 10 = 490
        assert state.my_qty == 490

    def test_pov_execution(self):
        state = LimitQueueState(
            strategy="pov",
            risk_limit=1000.0,
            orders_ahead=100,
            my_qty=500,
            is_filled=False,
            ofi_stream=[],
        )

        # 3. POV Execution (Queue ahead)
        _, state = execute_pov(50, state)
        assert state.orders_ahead == 50
        assert state.my_qty == 500

        # 4. POV Execution (Queue cleared)
        _, state = execute_pov(100, state)
        assert state.orders_ahead == 0
        assert state.my_qty == 450


class TestQuantInventoryClamped:
    def test_vwap_clamps_to_zero(self):
        state = LimitQueueState(orders_ahead=1, my_qty=5)
        _, state = execute_vwap(100, state)
        assert state.my_qty == 0

    def test_pov_clamps_to_zero(self):
        state = LimitQueueState(orders_ahead=1, my_qty=5)
        _, state = execute_pov(100, state)
        assert state.orders_ahead == 0
        assert state.my_qty == 0

    def test_passive_clamps_to_zero(self):
        state = LimitQueueState(orders_ahead=0, my_qty=5)
        _, state = execute_passive(100, state)
        assert state.my_qty == 0
