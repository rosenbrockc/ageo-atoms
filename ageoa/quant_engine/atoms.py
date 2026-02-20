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

    <!-- conceptual_profile -->
    {
        "abstract_name": "Paired Gradient Imbalance Tracker",
        "conceptual_transform": "Computes the net difference between two opposing magnitude vectors (ingress vs egress) and updates a temporal log of the cumulative imbalance. It maps two competing fluxes to a single scalar representation of pressure.",
        "abstract_inputs": [
            {
                "name": "bid_px",
                "description": "A scalar representing the current lower-bound threshold."
            },
            {
                "name": "bid_qty",
                "description": "A scalar representing the magnitude of the lower-bound flux."
            },
            {
                "name": "ask_px",
                "description": "A scalar representing the current upper-bound threshold."
            },
            {
                "name": "ask_qty",
                "description": "A scalar representing the magnitude of the upper-bound flux."
            },
            {
                "name": "trade_qty",
                "description": "A scalar representing the total realized flux in the current interval."
            },
            {
                "name": "state",
                "description": "A state object containing the historical record of imbalance measurements."
            }
        ],
        "abstract_outputs": [
            {
                "name": "ofi",
                "description": "A scalar representing the net imbalance in the current interval."
            },
            {
                "name": "new_state",
                "description": "The updated state object containing the new imbalance measurement."
            }
        ],
        "algorithmic_properties": [
            "differential-measure",
            "state-updating",
            "temporal-accumulation"
        ],
        "cross_disciplinary_applications": [
            "Monitoring the net pressure difference in a fluid pipeline system.",
            "Analyzing the imbalance between data ingress and egress in a network buffer.",
            "Measuring the thermal imbalance in a heat exchange system between source and sink."
        ]
    }
    <!-- /conceptual_profile -->
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

    <!-- conceptual_profile -->
    {
        "abstract_name": "Constant-Ratio Resource Depletor",
        "conceptual_transform": "Reduces a local state variable by a fixed proportion of an observed external flux, subject to available capacity. It ensures that the resource consumption rate scales linearly with external throughput.",
        "abstract_inputs": [
            {
                "name": "trade_qty",
                "description": "A scalar representing the magnitude of the external incoming flux."
            },
            {
                "name": "state",
                "description": "A state object containing the current local resource level."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple containing None and the updated state with reduced resource levels."
            }
        ],
        "algorithmic_properties": [
            "proportional-scaling",
            "capacity-constrained",
            "deterministic"
        ],
        "cross_disciplinary_applications": [
            "Automatically releasing a chemical reagent into a stream at a fixed percentage of the total flow rate.",
            "Proportionally scaling cloud computing resources based on incoming request volume.",
            "Distributing water from a reservoir based on a fixed percentage of current downstream demand."
        ]
    }
    <!-- /conceptual_profile -->
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

    <!-- conceptual_profile -->
    {
        "abstract_name": "Sequential Priority Resource Drain",
        "conceptual_transform": "Consumes an incoming flux to first deplete a primary buffer (priority queue) and then applies any remaining flux to deplete a secondary local resource. It implements a hierarchical fulfillment logic.",
        "abstract_inputs": [
            {
                "name": "trade_qty",
                "description": "A scalar representing the magnitude of the incoming flux."
            },
            {
                "name": "state",
                "description": "A state object containing the primary buffer level and the secondary resource level."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple containing None and the updated state with adjusted buffer and resource levels."
            }
        ],
        "algorithmic_properties": [
            "hierarchical-fulfillment",
            "priority-based",
            "sequential-depletion"
        ],
        "cross_disciplinary_applications": [
            "Clearing backordered customer shipments before fulfilling new internal inventory requests in logistics.",
            "Processing high-priority external tasks in a computing system before internal maintenance tasks.",
            "Allocating disaster relief supplies to primary impact zones before secondary aid regions."
        ]
    }
    <!-- /conceptual_profile -->
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

    <!-- conceptual_profile -->
    {
        "abstract_name": "Overflow-Triggered Resource Drain",
        "conceptual_transform": "Consumes a local resource only using the component of an incoming flux that exceeds a specified primary buffer capacity. It acts as a residual consumer of external volume.",
        "abstract_inputs": [
            {
                "name": "trade_qty",
                "description": "A scalar representing the magnitude of the incoming flux."
            },
            {
                "name": "state",
                "description": "A state object containing the primary buffer capacity and the local resource level."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple containing None and the updated state with adjusted resource levels."
            }
        ],
        "algorithmic_properties": [
            "residual-consumption",
            "overflow-based",
            "threshold-triggered"
        ],
        "cross_disciplinary_applications": [
            "Shunting excess current to a storage battery only when the primary grid demand is fully satisfied.",
            "Utilizing overflow in a water management system to fill secondary agricultural basins.",
            "Capturing waste heat for recovery only when the thermal output exceeds a specific operating threshold."
        ]
    }
    <!-- /conceptual_profile -->
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
