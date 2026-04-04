from __future__ import annotations

import icontract
import numpy as np
import pytest

from ageoa.pronto.foot_contact.atoms import mode_snapshot_readout
from ageoa.pronto.leg_odometer.atoms import posequeryaccessors, velocitystatereadout


def test_velocitystatereadout_returns_numpy_arrays() -> None:
    velocity, covariance = velocitystatereadout(
        {
            "xd_b": np.array([0.25, -0.5, 0.75], dtype=float),
            "vel_cov": np.diag([1.0, 2.0, 3.0]).astype(float),
        }
    )
    np.testing.assert_allclose(velocity, np.array([0.25, -0.5, 0.75], dtype=float))
    np.testing.assert_allclose(covariance, np.diag([1.0, 2.0, 3.0]).astype(float))


def test_posequeryaccessors_returns_identity_pose_defaults() -> None:
    result = posequeryaccessors()
    assert set(result) == {"position", "orientation"}
    np.testing.assert_allclose(result["position"], np.zeros(3, dtype=float))
    np.testing.assert_allclose(result["orientation"], np.eye(3, dtype=float))


def test_mode_snapshot_readout_returns_current_and_previous_modes() -> None:
    assert mode_snapshot_readout({"mode": "stance", "previous_mode": "swing"}) == ("stance", "swing")


def test_velocitystatereadout_rejects_missing_state() -> None:
    with pytest.raises(icontract.ViolationError):
        velocitystatereadout(None)  # type: ignore[arg-type]
