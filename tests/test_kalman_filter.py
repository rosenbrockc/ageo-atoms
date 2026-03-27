from __future__ import annotations

import pytest

from ageoa.institutional_quant_engine.kalman_filter.atoms import (
    kalmanfilterinit,
    kalmanmeasurementupdate,
)


def test_kalmanfilterinit_builds_scalar_state() -> None:
    state = kalmanfilterinit(0.1, 0.2, 1.0)

    assert state.x == 0.0
    assert state.p == 1.0
    assert state.q == 0.1
    assert state.r == 0.2


def test_kalmanmeasurementupdate_matches_scalar_kalman_equations() -> None:
    prior = kalmanfilterinit(0.1, 0.2, 1.0)

    posterior = kalmanmeasurementupdate(prior, 2.0)

    predicted_p = 1.0 + 0.1
    gain = predicted_p / (predicted_p + 0.2)
    expected_x = gain * 2.0
    expected_p = (1.0 - gain) * predicted_p

    assert posterior.x == pytest.approx(expected_x)
    assert posterior.p == pytest.approx(expected_p)
    assert posterior.q == prior.q
    assert posterior.r == prior.r
