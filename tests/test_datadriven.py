"""Tests for ageoa.datadriven atoms."""

import sys
import types
from importlib import util
from pathlib import Path

import numpy as np
import pytest


def _load_datadriven_atoms():
    root = Path(__file__).resolve().parents[1]
    ageoa_pkg = sys.modules.get("ageoa")
    if ageoa_pkg is None or not getattr(ageoa_pkg, "__path__", None):
        ageoa_pkg = types.ModuleType("ageoa")
        ageoa_pkg.__path__ = [str(root / "ageoa")]
        ageoa_pkg.__package__ = "ageoa"
        sys.modules["ageoa"] = ageoa_pkg
    datadriven_pkg = sys.modules.get("ageoa.datadriven")
    if datadriven_pkg is None or not getattr(datadriven_pkg, "__path__", None):
        datadriven_pkg = types.ModuleType("ageoa.datadriven")
        datadriven_pkg.__path__ = [str(root / "ageoa" / "datadriven")]
        datadriven_pkg.__package__ = "ageoa.datadriven"
        sys.modules["ageoa.datadriven"] = datadriven_pkg
    spec = util.spec_from_file_location("ageoa.datadriven.atoms", root / "ageoa" / "datadriven" / "atoms.py")
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules["ageoa.datadriven.atoms"] = module
    spec.loader.exec_module(module)
    return module


datadriven_atoms = _load_datadriven_atoms()
discover_equations = datadriven_atoms.discover_equations


class TestDiscoverEquations:
    def test_sindy_linear_system(self, monkeypatch):
        monkeypatch.setattr(datadriven_atoms, "_get_jl", lambda: (_ for _ in ()).throw(RuntimeError("julia unavailable")))
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

        assert len(res.equations) == 2
        assert len(res.parameter_map) == 2

        vals = list(res.parameter_map.values())
        assert any(np.isclose(v, -0.5, atol=1e-2) for v in vals)
        assert any(np.isclose(v, 0.8, atol=1e-2) for v in vals)
