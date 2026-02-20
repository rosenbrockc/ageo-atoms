"""Tests for ageoa.rust_robotics atoms."""

import numpy as np
import pytest
import icontract
import math

import ageoa.rust_robotics as ag_robotics


class TestPurePursuit:
    """Tests for the pure_pursuit atom."""

    def test_positive_basic(self):
        position_current = ag_robotics.controls.Point2D(x=0.0, y=0.0)
        position_target = ag_robotics.controls.Point2D(x=1.0, y=0.0)
        yaw_current = 0.0
        target_distance = 5.0
        wheelbase = 2.5
        
        result = ag_robotics.pure_pursuit(
            position_current, position_target, yaw_current, target_distance, wheelbase
        )
        assert np.isclose(result, 0.0, atol=1e-12)

    def test_positive_45_deg(self):
        position_current = ag_robotics.controls.Point2D(x=0.0, y=0.0)
        position_target = ag_robotics.controls.Point2D(x=1.0, y=1.0)
        yaw_current = 0.0
        target_distance = 5.0
        wheelbase = 2.5
        
        expected_rwa = math.atan2(np.sqrt(2)/2, 1)

        result = ag_robotics.pure_pursuit(
            position_current, position_target, yaw_current, target_distance, wheelbase
        )
        assert np.isclose(result, expected_rwa, atol=1e-12)

    def test_require_positive_target_distance(self):
        with pytest.raises(icontract.ViolationError, match="strictly positive"):
            ag_robotics.pure_pursuit(ag_robotics.controls.Point2D(x=0.0, y=0.0), ag_robotics.controls.Point2D(x=1.0, y=0.0), 0.0, 0.0, 2.5)

    def test_require_positive_wheelbase(self):
        with pytest.raises(icontract.ViolationError, match="strictly positive"):
            ag_robotics.pure_pursuit(ag_robotics.controls.Point2D(x=0.0, y=0.0), ag_robotics.controls.Point2D(x=1.0, y=0.0), 0.0, 5.0, -1.0)


class TestRk4:
    """Tests for the rk4 atom."""

    def test_positive_cv(self):
        # Constant velocity dx/dt = v -> dx = v dt
        # State: x = [pos, vel]
        def func(x: np.ndarray, t: float) -> np.ndarray:
            A = np.array([[0.0, 1.0], [0.0, 0.0]])
            return A @ x

        x0 = np.array([0.0, 2.0])
        t0 = 0.0
        tf = 10.0
        
        result = ag_robotics.rk4(func, x0, t0, tf)
        expected = np.array([20.0, 2.0])
        assert np.allclose(result, expected, atol=1e-12)

    def test_require_1d_vector(self):
        def func(x: np.ndarray, t: float) -> np.ndarray:
            return x

        x0 = np.array([[0.0, 2.0]])
        with pytest.raises(icontract.ViolationError, match="1D vector"):
            ag_robotics.rk4(func, x0, 0.0, 10.0)

    def test_require_tf_greater_t0(self):
        def func(x: np.ndarray, t: float) -> np.ndarray:
            return x

        x0 = np.array([0.0, 2.0])
        with pytest.raises(icontract.ViolationError, match="tf must be strictly greater than t0"):
            ag_robotics.rk4(func, x0, 10.0, 0.0)
