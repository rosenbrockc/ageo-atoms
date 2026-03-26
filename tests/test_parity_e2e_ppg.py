"""Auto-generated parity tests for e2e_ppg atoms.

DO NOT EDIT — regenerate with: python scripts/generate_parity_tests.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# Import helpers from test_parity without requiring tests to be a package
_spec = importlib.util.spec_from_file_location(
    "test_parity", Path(__file__).parent / "test_parity.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_assert_outputs_match = _mod._assert_outputs_match
_deserialize = _mod._deserialize
_import_atom = _mod._import_atom

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "e2e_ppg"

class TestSignalarraynormalization:
    FIXTURE = FIXTURES_DIR / "kazemi_wrapper/signalarraynormalization.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("e2e_ppg/kazemi_wrapper:signalarraynormalization")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestNormalizesignal:
    FIXTURE = FIXTURES_DIR / "kazemi_wrapper_d12/normalizesignal.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(1))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("e2e_ppg/kazemi_wrapper_d12:normalizesignal")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

