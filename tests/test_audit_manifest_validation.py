from __future__ import annotations

from pathlib import Path

from auditlib import manifest_validation
from auditlib.inventory import build_manifest
from auditlib.manifest_validation import validate_manifest


def _manifest_payload(atom: dict, *, atom_count: int = 1) -> dict:
    return {
        "schema_version": "1.0",
        "metadata": {
            "generated_at": "2026-03-25T00:00:00+00:00",
            "repo": "ageo-atoms",
            "generator": "tests",
            "phase": "phase-5",
        },
        "summary": {
            "atom_count": atom_count,
            "inventory_error_count": 0,
            "family_counts": {},
            "source_kind_counts": {},
            "risk_tier_counts": {},
            "unmapped_upstream_count": 0,
        },
        "atoms": [atom],
        "inventory_errors": [],
    }


def _heuristic_atom(tmp_path: Path, *, docstring_summary: str | None) -> dict:
    wrapper_path = tmp_path / "ageoa" / "example.py"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text("def example(x: int) -> int:\n    return x\n")
    return {
        "atom_id": "ageoa.example.example@ageoa/example.py:1",
        "atom_key": "example:example",
        "atom_name": "ageoa.example.example",
        "module_path": "ageoa/example.py",
        "wrapper_symbol": "example",
        "wrapper_line": 1,
        "source_kind": "generated_ingest",
        "risk_tier": "medium",
        "stateful_kind": "none",
        "stochastic": False,
        "procedural": False,
        "authoritative_sources": [],
        "risk_reasons": [],
        "status_basis": {},
        "docstring_summary": docstring_summary,
        "has_docstring": bool(docstring_summary),
        "has_weak_types": False,
        "skeleton": False,
    }


def test_validate_manifest_accepts_current_payload() -> None:
    manifest = build_manifest()
    report = validate_manifest(manifest)
    assert report["ok"] is True
    assert report["summary"]["error_count"] == 0
    assert report["summary"]["heuristic_atom_count"] > 0
    assert report["summary"]["heuristic_error_count"] == 0


def test_validate_manifest_detects_duplicate_atom_ids() -> None:
    manifest = build_manifest()
    manifest["atoms"][1]["atom_id"] = manifest["atoms"][0]["atom_id"]
    report = validate_manifest(manifest)
    codes = {finding["code"] for finding in report["findings"]}
    assert report["ok"] is False
    assert "MANIFEST_DUPLICATE_ATOM_ID" in codes


def test_validate_manifest_reports_heuristic_dejargonization_warnings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manifest_validation, "ROOT", tmp_path)
    payload = _manifest_payload(_heuristic_atom(tmp_path, docstring_summary="Time."))
    report = validate_manifest(payload)
    codes = {finding["code"] for finding in report["findings"]}
    assert report["ok"] is True
    assert "HEURISTIC_DEJARGONIZATION_WEAK" in codes
    assert "HEURISTIC_DOCSTRING_THIN" in codes
    assert report["summary"]["heuristic_warning_count"] >= 1


def test_validate_manifest_rejects_missing_or_placeholder_heuristic_docstrings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manifest_validation, "ROOT", tmp_path)
    payload = _manifest_payload(_heuristic_atom(tmp_path, docstring_summary="Derived deterministically from inputs."))
    report = validate_manifest(payload)
    codes = {finding["code"] for finding in report["findings"]}
    assert report["ok"] is False
    assert "HEURISTIC_DOCSTRING_PLACEHOLDER" in codes


def test_validate_manifest_rejects_missing_heuristic_docstrings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manifest_validation, "ROOT", tmp_path)
    payload = _manifest_payload(_heuristic_atom(tmp_path, docstring_summary=None))
    report = validate_manifest(payload)
    codes = {finding["code"] for finding in report["findings"]}
    assert report["ok"] is False
    assert "HEURISTIC_DOCSTRING_MISSING" in codes
