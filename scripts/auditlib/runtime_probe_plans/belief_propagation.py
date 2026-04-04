"""Runtime probe plans for belief propagation families."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


def _assert_initialized_bp_state(result: Any) -> None:
    state_in, state_out = result
    assert state_in.t == 0
    assert state_out.t == 0
    assert state_in.pgm is state_out.pgm
    assert set(state_in.msg.keys()) == {"x", "f"}
    assert set(state_in.msg["x"].keys()) == {"f"}
    assert set(state_in.msg["f"].keys()) == {"x"}
    np.testing.assert_allclose(state_in.msg["x"]["f"], np.array([0.5, 0.5], dtype=float))
    np.testing.assert_allclose(state_in.msg["f"]["x"], np.array([0.5, 0.5], dtype=float))


def _assert_bp_belief_query_result(result: Any) -> None:
    (belief, state_out), returned_state = result
    assert state_out.t == 2
    assert returned_state.t == 2
    np.testing.assert_allclose(np.asarray(belief, dtype=float).sum(), 1.0, atol=1e-8)
    assert tuple(np.asarray(belief).shape) == (2,)


def get_probe_plans() -> dict[str, object]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan

    pgm = nx.DiGraph()
    pgm.add_node("x", card=2, potential=np.array([0.8, 0.2], dtype=float))
    pgm.add_node("f", card=2)
    pgm.add_edge("x", "f")
    pgm.add_edge("f", "x")

    def _build_state() -> Any:
        from ageoa.belief_propagation.loopy_bp.state_models import BPState

        return BPState()

    return {
        "ageoa.belief_propagation.loopy_bp.initialize_message_passing_state": ProbePlan(
            positive=ProbeCase(
                "initialize_message_passing_state creates normalized bidirectional messages",
                lambda func: func(pgm, _build_state()),
                _assert_initialized_bp_state,
            ),
            negative=ProbeCase(
                "reject a missing probabilistic graph model",
                lambda func: func(None, _build_state()),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.belief_propagation.loopy_bp.run_loopy_message_passing_and_belief_query": ProbePlan(
            positive=ProbeCase(
                "run_loopy_message_passing_and_belief_query returns a normalized queried belief",
                lambda func: (
                    lambda initialized: func(initialized[0], "x", 2, initialized[1])
                )(
                    __import__("ageoa.belief_propagation.loopy_bp.atoms", fromlist=["initialize_message_passing_state"])
                    .initialize_message_passing_state(pgm, _build_state())
                ),
                _assert_bp_belief_query_result,
            ),
            negative=ProbeCase(
                "reject an unknown variable name during belief query",
                lambda func: (
                    lambda initialized: func(initialized[0], "missing", 1, initialized[1])
                )(
                    __import__("ageoa.belief_propagation.loopy_bp.atoms", fromlist=["initialize_message_passing_state"])
                    .initialize_message_passing_state(pgm, _build_state())
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
