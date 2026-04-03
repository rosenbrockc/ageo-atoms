"""Runtime probe plans for quant_engine families."""

from __future__ import annotations

from typing import Any, Callable


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    safe_import_module = rt.safe_import_module

    def _assert_calculate_ofi_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        ofi, state = result
        assert abs(float(ofi) - 3.0) < 1e-12
        assert list(state.ofi_stream or []) == [3.0]

    def _assert_queue_state(*, my_qty: int, orders_ahead: int | None) -> Callable[[Any], None]:
        def _assert(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            marker, state = result
            assert marker is None
            assert (state.my_qty or 0) == my_qty
            if orders_ahead is None:
                assert state.orders_ahead is None
            else:
                assert (state.orders_ahead or 0) == orders_ahead

        return _assert

    return {
        "ageoa.quant_engine.calculate_ofi": ProbePlan(
            positive=ProbeCase(
                "compute a deterministic order-flow imbalance and append it to state",
                lambda func: func(
                    100.0,
                    10,
                    101.0,
                    4,
                    3,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(ofi_stream=[]),
                ),
                _assert_calculate_ofi_bundle,
            ),
            negative=ProbeCase(
                "reject a negative bid quantity",
                lambda func: func(
                    100.0,
                    -1,
                    101.0,
                    4,
                    3,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(ofi_stream=[]),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quant_engine.execute_vwap": ProbePlan(
            positive=ProbeCase(
                "apply a deterministic vwap participation fill to inventory",
                lambda func: func(
                    20,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=9),
                ),
                _assert_queue_state(my_qty=7, orders_ahead=None),
            ),
            negative=ProbeCase(
                "reject a non-positive trade quantity for vwap execution",
                lambda func: func(
                    0,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=9),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quant_engine.execute_pov": ProbePlan(
            positive=ProbeCase(
                "apply a deterministic pov fill against queue priority",
                lambda func: func(
                    5,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                _assert_queue_state(my_qty=8, orders_ahead=0),
            ),
            negative=ProbeCase(
                "reject a non-positive trade quantity for pov execution",
                lambda func: func(
                    0,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quant_engine.execute_passive": ProbePlan(
            positive=ProbeCase(
                "apply a deterministic passive fill against queue priority",
                lambda func: func(
                    5,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                _assert_queue_state(my_qty=8, orders_ahead=0),
            ),
            negative=ProbeCase(
                "reject a non-positive trade quantity for passive execution",
                lambda func: func(
                    0,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
