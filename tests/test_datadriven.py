"""Tests for ageoa.datadriven atoms."""

import numpy as np
import pytest
from ageoa.datadriven.atoms import discover_equations


class TestDiscoverEquations:
    def test_sindy_linear_system(self):
        # Construct a simple linear system: dx/dt = -0.5 * x, dy/dt = 0.8 * y
        t = np.linspace(0, 10, 100)
        X = np.zeros((2, 100))
        Y = np.zeros((2, 100))

        # Fake data
        X[0, :] = np.exp(-0.5 * t)
        X[1, :] = np.exp(0.8 * t)

        Y[0, :] = -0.5 * X[0, :]
        Y[1, :] = 0.8 * X[1, :]

        res = discover_equations(
            X=X,
            Y=Y,
            variable_names=["x", "y"],
            max_degree=1,
            lambda_val=0.01
        )

        assert len(res.parameter_map) == 2

        vals = list(res.parameter_map.values())
        assert any(np.isclose(v, -0.5, atol=1e-2) for v in vals)
        assert any(np.isclose(v, 0.8, atol=1e-2) for v in vals)
