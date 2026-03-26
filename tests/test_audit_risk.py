from __future__ import annotations

import csv

from auditlib.risk import compute_atom_risk, run_risk_triage


def _base_record() -> dict:
    return {
        "atom_id": "ageoa.example.atom@ageoa/example.py:1",
        "atom_name": "ageoa.example.atom",
        "atom_key": "example:atom",
        "domain_family": "example",
        "wrapper_symbol": "atom",
        "structural_status": "pass",
        "structural_findings": [],
        "has_weak_types": False,
        "has_parity_tests": True,
        "review_basis_at": "2026-03-25T00:00:00+00:00",
        "source_revision": "abc123",
        "upstream_version": None,
        "runtime_status": "pass",
        "stateful": False,
        "stateful_kind": "none",
        "procedural": False,
        "source_kind": "hand_written",
        "ffi": False,
        "stochastic": False,
        "docstring_summary": "Compute a stable upstream operation.",
    }


def test_compute_atom_risk_for_structural_failure() -> None:
    record = _base_record()
    record["structural_status"] = "fail"
    findings = [{"code": "STRUCT_STUB_PUBLIC_API"}]
    result = compute_atom_risk(record, findings, signature=None, acceptability=None)
    assert result["risk_tier"] == "high"
    assert "RISK_STRUCTURAL_FAIL" in result["risk_reasons"]
    assert "RISK_STUB_PUBLIC_API" in result["risk_reasons"]
    assert result["review_priority"] == "review_now"


def test_compute_atom_risk_for_mapped_low_risk_atom() -> None:
    record = _base_record()
    signature = {
        "mapping_found": True,
        "upstream_mapping": {"function": "atom", "module": "example.module"},
        "upstream_signature": {"parameter_names": ["x"]},
        "findings": [],
    }
    acceptability = {"major_penalties": ""}
    result = compute_atom_risk(record, [], signature, acceptability)
    assert result["risk_tier"] == "low"
    assert result["review_priority"] == "review_later"


def test_compute_atom_risk_for_unmapped_generated_wrapper() -> None:
    record = _base_record()
    record["source_kind"] = "generated_ingest"
    record["has_parity_tests"] = False
    record["runtime_status"] = "unknown"
    result = compute_atom_risk(record, [], signature=None, acceptability={"major_penalties": "RUNTIME_NO_PROBE_EVIDENCE"})
    assert result["risk_tier"] in {"medium", "high"}
    assert "RISK_UPSTREAM_UNMAPPED" in result["risk_reasons"]
    assert "RISK_GENERATED_INGEST" in result["risk_reasons"]
    assert "RISK_MISSING_PARITY" in result["risk_reasons"]


def test_run_risk_triage_updates_manifest_and_writes_queue(tmp_path, monkeypatch) -> None:
    from auditlib import risk as risk_mod

    manifest_path = tmp_path / "audit_manifest.json"
    report_path = tmp_path / "risk_report.json"
    queue_path = tmp_path / "review_queue.csv"
    manifest = {
        "summary": {},
        "atoms": [
            _base_record(),
            {
                **_base_record(),
                "atom_id": "ageoa.example.generated@ageoa/example.py:2",
                "atom_name": "ageoa.example.generated",
                "atom_key": "example:generated",
                "wrapper_symbol": "generated",
                "source_kind": "generated_ingest",
                "has_parity_tests": False,
                "runtime_status": "unknown",
            },
        ],
    }
    manifest_path.write_text("{}\n")
    monkeypatch.setattr(risk_mod, "AUDIT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(risk_mod, "AUDIT_RISK_REPORT_PATH", report_path)
    monkeypatch.setattr(risk_mod, "AUDIT_REVIEW_QUEUE_CSV_PATH", queue_path)
    updated_manifest, report = run_risk_triage(manifest)
    assert len(updated_manifest["atoms"]) == 2
    assert report["summary"]["atom_count"] == 2
    assert report_path.exists()
    assert queue_path.exists()
    with queue_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["atom_name"] == "ageoa.example.generated"
