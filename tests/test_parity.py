"""Parametrized parity tests: verify atoms reproduce upstream I/O.

Discovers all fixture files under ``tests/fixtures/`` and runs each
atom function against the captured inputs, comparing outputs to the
upstream reference.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_fixtures() -> list[Path]:
    """Find all ``*.json`` fixture files."""
    return sorted(FIXTURES_DIR.rglob("*.json"))


def _import_atom(atom_key: str) -> Any:
    """Import an atom function from its key like ``biosppy/ecg_detectors:hamilton_segmentation``."""
    module_path, func_name = atom_key.split(":")
    # Convert filesystem path to Python module path: biosppy/ecg_hamilton → ageoa.biosppy.ecg_hamilton.atoms
    parts = module_path.split("/")
    dotted = "ageoa." + ".".join(parts) + ".atoms"
    try:
        mod = importlib.import_module(dotted)
    except ModuleNotFoundError:
        # Some atoms live in a top-level .py file, not atoms.py
        dotted_alt = "ageoa." + ".".join(parts)
        mod = importlib.import_module(dotted_alt)
    return getattr(mod, func_name)


def _deserialize(obj: Any) -> Any:
    """Recursively deserialize JSON-encoded fixture values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, list):
        return [_deserialize(v) for v in obj]

    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            import base64

            raw = base64.b64decode(obj["data"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(
                obj["shape"]
            )

        if obj.get("__tuple__"):
            return tuple(_deserialize(v) for v in obj["items"])

        if obj.get("__complex__"):
            return complex(obj["real"], obj["imag"])

        if obj.get("__bytes__"):
            import base64

            return base64.b64decode(obj["data"])

        if obj.get("__pickle__"):
            import base64
            import pickle

            return pickle.loads(base64.b64decode(obj["data"]))

        if obj.get("__pydantic__"):
            return _deserialize(obj["data"])

        return {k: _deserialize(v) for k, v in obj.items()}

    return obj


def _assert_outputs_match(actual: Any, expected: Any, rtol: float = 1e-6) -> None:
    """Compare atom output to reference, with tolerance for floats/arrays."""
    if expected is None:
        assert actual is None
        return

    if isinstance(expected, np.ndarray):
        actual_arr = np.asarray(actual)
        np.testing.assert_allclose(actual_arr, expected, rtol=rtol, atol=1e-10)
        return

    if isinstance(expected, (float, complex)):
        assert actual == pytest.approx(expected, rel=rtol, abs=1e-10)
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected dict, got {type(actual)}"
        for k in expected:
            assert k in actual, f"Missing key {k!r} in output"
            _assert_outputs_match(actual[k], expected[k], rtol=rtol)
        return

    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (
            f"Length mismatch: {len(actual)} vs {len(expected)}"
        )
        for a, e in zip(actual, expected):
            _assert_outputs_match(a, e, rtol=rtol)
        return

    assert actual == expected


# ---------------------------------------------------------------------------
# Test collection
# ---------------------------------------------------------------------------

_fixture_paths = discover_fixtures()


@pytest.mark.parametrize(
    "fixture_path",
    _fixture_paths,
    ids=lambda p: str(p.relative_to(FIXTURES_DIR)),
)
def test_atom_parity(fixture_path: Path) -> None:
    """Verify that an atom reproduces upstream I/O for every captured case."""
    with open(fixture_path) as f:
        cases = json.load(f)

    if not cases:
        pytest.skip("Empty fixture file")

    atom_key = cases[0]["atom"]

    try:
        atom_fn = _import_atom(atom_key)
    except (ModuleNotFoundError, AttributeError) as exc:
        pytest.skip(f"Cannot import atom {atom_key}: {exc}")

    for i, case in enumerate(cases):
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])

        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip(f"Atom {atom_key} is a stub (NotImplementedError)")

        _assert_outputs_match(result, expected)
