"""Tests for the uncertainty measurement harness (Phase 3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_base_inputs import get_base_input, BASE_INPUT_REGISTRY
from scripts.measure_uncertainty import measure_atom, write_uncertainty_json


# ---------------------------------------------------------------------------
# generate_base_inputs
# ---------------------------------------------------------------------------


class TestGenerateBaseInputs:
    def test_known_atom_returns_array(self):
        x = get_base_input("fft")
        assert isinstance(x, np.ndarray)
        assert x.ndim >= 1
        assert len(x) > 0

    def test_unknown_atom_returns_fallback(self):
        x = get_base_input("totally_unknown_atom_xyz")
        assert isinstance(x, np.ndarray)
        assert x.shape == (256,)

    def test_deterministic_fallback(self):
        a = get_base_input("test_atom_abc")
        b = get_base_input("test_atom_abc")
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# measure_atom — smoke tests
# ---------------------------------------------------------------------------


class TestMeasureAtom:
    def test_identity_factor_near_one(self):
        """An identity function should have factor ~1.0."""
        def identity(x: np.ndarray) -> np.ndarray:
            return x.copy()

        result = measure_atom("identity", identity, seed=42)
        assert result is not None
        assert result["mode"] == "empirical"
        assert 0.5 < result["scalar_factor"] < 2.0
        assert result["confidence"] > 0.0
        assert result["n_trials"] > 0

    def test_fft_factor_in_range(self):
        """FFT should have a reasonable error expansion factor."""
        result = measure_atom("fft", np.fft.fft, seed=42)
        assert result is not None
        # FFT can amplify noise by O(sqrt(n)) so factor up to ~20 is plausible
        assert 0.5 < result["scalar_factor"] < 50.0
        assert result["confidence"] > 0.3

    def test_hostile_atom_skipped(self):
        """An atom that fails >50% of the time should return None."""
        call_count = 0

        def hostile(x: np.ndarray) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count % 5 != 0:  # 80% failure rate
                raise ValueError("hostile failure")
            return x

        result = measure_atom("hostile", hostile, seed=42)
        # With 80% failure rate across all epsilons, should be None
        assert result is None

    def test_deterministic_seeding(self):
        """Same seed should give the same result."""
        a = measure_atom("fft", np.fft.fft, seed=12345)
        b = measure_atom("fft", np.fft.fft, seed=12345)
        assert a is not None and b is not None
        assert a["scalar_factor"] == b["scalar_factor"]
        assert a["n_trials"] == b["n_trials"]


# ---------------------------------------------------------------------------
# write_uncertainty_json — schema validation
# ---------------------------------------------------------------------------


class TestWriteUncertaintyJson:
    def test_creates_valid_json(self, tmp_path):
        estimate = {
            "mode": "empirical",
            "scalar_factor": 1.5,
            "confidence": 0.85,
            "n_trials": 450,
            "epsilon": 1e-4,
            "input_regime": "standard_normal((256,))",
            "notes": "test",
        }
        path = write_uncertainty_json("fft", tmp_path, estimate)
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["atom"] == "fft"
        assert len(data["estimates"]) == 1
        assert data["estimates"][0]["mode"] == "empirical"
        assert data["estimates"][0]["scalar_factor"] > 0
        assert 0 <= data["estimates"][0]["confidence"] <= 1

    def test_merges_with_existing(self, tmp_path):
        """Writing a new empirical estimate should replace old empirical, keep others."""
        existing = {
            "atom": "fft",
            "estimates": [
                {"mode": "heuristic", "scalar_factor": 1.5, "confidence": 0.2},
                {"mode": "empirical", "scalar_factor": 1.3, "confidence": 0.7},
            ],
        }
        uj_path = tmp_path / "uncertainty.json"
        uj_path.write_text(json.dumps(existing))

        new_estimate = {
            "mode": "empirical",
            "scalar_factor": 1.55,
            "confidence": 0.9,
            "n_trials": 500,
            "epsilon": 1e-4,
            "input_regime": "test",
            "notes": "updated",
        }
        write_uncertainty_json("fft", tmp_path, new_estimate)

        data = json.loads(uj_path.read_text())
        assert len(data["estimates"]) == 2  # heuristic + new empirical
        modes = [e["mode"] for e in data["estimates"]]
        assert "heuristic" in modes
        assert "empirical" in modes
        # The empirical one should be the new one
        emp = [e for e in data["estimates"] if e["mode"] == "empirical"][0]
        assert emp["scalar_factor"] == 1.55
