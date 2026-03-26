"""Tests for rust_robotics."""

import pytest
import numpy as np
import icontract
from ageoa.rust_robotics.atoms import n_joint_arm_solver, dijkstra_path_planning


class TestNJointArmSolver:
    def test_returns_result(self):
        result = n_joint_arm_solver(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            n_joint_arm_solver(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            n_joint_arm_solver(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            n_joint_arm_solver(np.array([np.nan]))


class TestDijkstraPathPlanning:
    def test_returns_result(self):
        adj = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]])
        result = dijkstra_path_planning(adj)
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            dijkstra_path_planning(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            dijkstra_path_planning(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            dijkstra_path_planning(np.array([np.inf]))
