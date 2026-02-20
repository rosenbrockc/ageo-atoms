"""Tests for ageoa.pronto atoms."""

import numpy as np
import pytest
import icontract

import ageoa.pronto as ag_pronto


class TestProntoAtoms:
    """Tests for the Pronto EKF and Contact Gating atoms."""

    def test_ekf_update(self):
        # 21 element vector initialized to 0
        vec = np.zeros(21)
        # Identity quaternion [w, x, y, z]
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        utime = 1000

        state = ag_pronto.EKFState(vec=vec, quat=quat, utime=utime)
        
        gyro = np.array([0.1, 0.0, 0.0])
        accel = np.array([0.0, 0.0, 9.81])
        dt = 0.01

        # We're updating the state, and we should expect some changes in velocity/orientation
        new_state = ag_pronto.ekf_update(state, gyro, accel, dt)

        assert new_state.vec.shape == (21,)
        assert new_state.quat.shape == (4,)
        # Check if the state actually evolved
        assert not np.allclose(new_state.vec, np.zeros(21))

    def test_ekf_update_requires_21_elements(self):
        vec = np.zeros(20)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        utime = 1000

        state = ag_pronto.EKFState(vec=vec, quat=quat, utime=utime)
        gyro = np.array([0.1, 0.0, 0.0])
        accel = np.array([0.0, 0.0, 9.81])
        dt = 0.01

        with pytest.raises(icontract.ViolationError):
            ag_pronto.ekf_update(state, gyro, accel, dt)

    def test_contact_gating(self):
        # Instantiate a C++ classifier
        ptr = ag_pronto.contact_classifier_create()
        assert ptr != 0

        # Update walking phase with some contact conditions
        mode = ag_pronto.contact_classifier_update(
            classifier_ptr=ptr,
            utime=1000,
            left_contact=True,
            right_contact=True,
            left_contact_strong=True,
            right_contact_strong=True
        )
        
        # We expect a valid mode out of the classifier
        assert isinstance(mode, int)

        # Cleanup
        ag_pronto.contact_classifier_destroy(ptr)
