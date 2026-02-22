"""Tests for molecular_docking."""

import pytest
import numpy as np
import icontract
from ageoa.molecular_docking.atoms import quantum_mwis_solver, greedy_lattice_mapping


class TestQuantumMWISSolver:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            quantum_mwis_solver(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            quantum_mwis_solver(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            quantum_mwis_solver(np.array([]))


class TestGreedyLatticeMapping:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            greedy_lattice_mapping(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            greedy_lattice_mapping(None)

    def test_precondition_wrong_ndim(self):
        with pytest.raises(icontract.ViolationError):
            greedy_lattice_mapping(np.array([1.0, 2.0]))
