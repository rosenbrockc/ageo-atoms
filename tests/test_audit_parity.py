from __future__ import annotations

import csv
import json

from auditlib.parity import derive_parity_entry, inventory_parity_fixtures, run_parity_coverage


def _base_record() -> dict:
    return {
        "atom_id": "ageoa.example.atom@ageoa/example.py:1",
        "atom_name": "ageoa.example.atom",
        "atom_key": "example:atom",
        "domain_family": "example",
        "risk_tier": "medium",
        "review_priority": "review_soon",
        "runtime_status": "unknown",
        "stateful": False,
        "ffi": False,
        "source_kind": "hand_written",
        "overall_verdict": "acceptable_with_limits",
        "structural_findings": [],
        "status_basis": {},
    }


def test_inventory_parity_fixtures_counts_positive_and_negative_cases(tmp_path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "atom.json").write_text(
        json.dumps(
            [
                {"atom": "example:atom", "inputs": {"x": 1}, "output": 2},
                {
                    "atom": "example:atom",
                    "inputs": {"x": -1},
                    "expected_exception": "ValueError",
                },
            ]
        )
    )
    inventory = inventory_parity_fixtures(fixtures_dir)
    atom_entry = inventory["atoms"]["example:atom"]
    assert inventory["fixture_file_count"] == 1
    assert inventory["total_case_count"] == 2
    assert atom_entry["fixture_file_count"] == 1
    assert atom_entry["case_count"] == 2
    assert atom_entry["positive_case_count"] == 1
    assert atom_entry["negative_case_count"] == 1
    assert atom_entry["has_usage_equivalence"] is True


def test_derive_parity_entry_prefers_fixture_usage_equivalence() -> None:
    entry = derive_parity_entry(
        _base_record(),
        fixture_entry={
            "fixture_file_count": 1,
            "case_count": 2,
            "positive_case_count": 2,
            "negative_case_count": 0,
            "empty_fixture_count": 0,
            "has_usage_equivalence": True,
            "fixture_paths": ["example/atom.json"],
        },
        probe={"status": "pass", "positive_pass": True, "negative_pass": False, "parity_used": False},
    )
    assert entry["parity_coverage_level"] == "parity_or_usage_equivalent"
    assert "PARITY_FIXTURE_PRESENT" in entry["parity_coverage_reasons"]
    assert "PARITY_USAGE_EQUIVALENCE_PRESENT" in entry["parity_coverage_reasons"]
    assert entry["usage_test_coverage"] == "positive_path"


def test_derive_parity_entry_uses_negative_probe_for_contract_coverage() -> None:
    entry = derive_parity_entry(
        _base_record(),
        fixture_entry=None,
        probe={"status": "pass", "positive_pass": True, "negative_pass": True, "parity_used": False},
    )
    assert entry["parity_coverage_level"] == "positive_and_negative"
    assert "PARITY_RUNTIME_PROBE_SUPPORT" in entry["parity_coverage_reasons"]
    assert "PARITY_NEGATIVE_CASES_PRESENT" in entry["parity_coverage_reasons"]
    assert "PARITY_MISSING_USAGE_EQUIVALENCE" in entry["parity_coverage_reasons"]
    assert entry["usage_test_coverage"] == "positive_and_negative"


def test_run_parity_coverage_updates_manifest_and_writes_outputs(tmp_path, monkeypatch) -> None:
    from auditlib import parity as parity_mod

    fixtures_dir = tmp_path / "fixtures"
    probes_dir = tmp_path / "probes"
    audit_dir = tmp_path / "audit"
    manifest_path = tmp_path / "audit_manifest.json"
    fixtures_dir.mkdir()
    probes_dir.mkdir()
    audit_dir.mkdir()

    (fixtures_dir / "atom.json").write_text(
        json.dumps([{"atom": "example:atom", "inputs": {"x": 1}, "output": 2}])
    )
    (probes_dir / "probe.json").write_text(
        json.dumps(
            {
                "atom_id": "ageoa.example.atom@ageoa/example.py:1",
                "status": "pass",
                "probe_status": "executed",
                "parity_used": False,
                "positive_probe": {"status": "pass"},
                "negative_probe": {"status": "fail"},
                "findings": ["RUNTIME_PROBE_PASS"],
            }
        )
    )
    manifest = {"summary": {}, "atoms": [_base_record()]}
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(parity_mod, "FIXTURES_DIR", fixtures_dir)
    monkeypatch.setattr(parity_mod, "AUDIT_PROBES_DIR", probes_dir)
    monkeypatch.setattr(parity_mod, "AUDIT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(parity_mod, "AUDIT_PARITY_COVERAGE_PATH", audit_dir / "parity_coverage.json")
    monkeypatch.setattr(parity_mod, "AUDIT_PARITY_COVERAGE_CSV_PATH", audit_dir / "parity_coverage.csv")
    monkeypatch.setattr(parity_mod, "AUDIT_PARITY_BACKLOG_PATH", audit_dir / "parity_backlog.json")

    updated_manifest, report, backlog = run_parity_coverage(manifest)
    atom = updated_manifest["atoms"][0]
    assert atom["parity_coverage_level"] == "parity_or_usage_equivalent"
    assert atom["parity_fixture_count"] == 1
    assert atom["parity_case_count"] == 1
    assert atom["has_parity_tests"] is True
    assert atom["parity_test_status"] == "pass"
    assert report["by_level"]["parity_or_usage_equivalent"] == 1
    assert backlog["summary"]["missing_count"] == 0

    with (audit_dir / "parity_coverage.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["parity_coverage_level"] == "parity_or_usage_equivalent"


def test_run_parity_coverage_maps_probe_only_positive_path_to_partial(tmp_path, monkeypatch) -> None:
    from auditlib import parity as parity_mod

    fixtures_dir = tmp_path / "fixtures"
    probes_dir = tmp_path / "probes"
    audit_dir = tmp_path / "audit"
    manifest_path = tmp_path / "audit_manifest.json"
    fixtures_dir.mkdir()
    probes_dir.mkdir()
    audit_dir.mkdir()

    (probes_dir / "probe.json").write_text(
        json.dumps(
            {
                "atom_id": "ageoa.example.atom@ageoa/example.py:1",
                "status": "pass",
                "probe_status": "executed",
                "parity_used": False,
                "positive_probe": {"status": "pass"},
                "negative_probe": {"status": "fail"},
                "findings": ["RUNTIME_PROBE_PASS"],
            }
        )
    )
    manifest = {"summary": {}, "atoms": [_base_record()]}
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(parity_mod, "FIXTURES_DIR", fixtures_dir)
    monkeypatch.setattr(parity_mod, "AUDIT_PROBES_DIR", probes_dir)
    monkeypatch.setattr(parity_mod, "AUDIT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(parity_mod, "AUDIT_PARITY_COVERAGE_PATH", audit_dir / "parity_coverage.json")
    monkeypatch.setattr(parity_mod, "AUDIT_PARITY_COVERAGE_CSV_PATH", audit_dir / "parity_coverage.csv")
    monkeypatch.setattr(parity_mod, "AUDIT_PARITY_BACKLOG_PATH", audit_dir / "parity_backlog.json")

    updated_manifest, _, _ = run_parity_coverage(manifest)
    atom = updated_manifest["atoms"][0]
    assert atom["parity_coverage_level"] == "positive_path"
    assert atom["has_parity_tests"] is True
    assert atom["parity_test_status"] == "partial"
