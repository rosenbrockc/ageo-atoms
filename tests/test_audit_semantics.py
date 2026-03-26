from __future__ import annotations

from auditlib.acceptability import score_acceptability
from auditlib import semantics


def test_semantic_report_rolls_findings_into_manifest(monkeypatch) -> None:
    manifest = {
        "summary": {},
        "atoms": [
            {
                "atom_id": "atom",
                "atom_name": "ageoa.example.atom",
                "module_path": "ageoa/example.py",
                "wrapper_line": 1,
                "blocking_findings": [],
                "required_actions": [],
                "status_basis": {},
                "has_docstring": True,
            }
        ],
    }
    monkeypatch.setattr(
        semantics,
        "load_semantic_sections",
        lambda _atom_id: {
            "runtime_probe": {"status": "pass", "findings": ["RUNTIME_PROBE_PASS"], "notes": [], "source_refs": []},
            "return_fidelity": {"status": "fail", "findings": ["RETURN_FABRICATED_ATTRIBUTE"], "notes": [], "source_refs": []},
            "generated_nouns": {"status": "partial", "findings": ["NOUN_UNDOCUMENTED_OUTPUT"], "notes": [], "source_refs": []},
        },
    )
    updated_manifest, report = semantics.build_semantic_report(manifest, {})
    atom = updated_manifest["atoms"][0]
    assert atom["runtime_status"] == "pass"
    assert atom["semantic_status"] == "fail"
    assert atom["developer_semantics_status"] == "partial"
    assert "RETURN_FABRICATED_ATTRIBUTE" in atom["blocking_findings"]
    assert report["summary"]["runtime_status_counts"]["pass"] == 1


def test_acceptability_uses_semantic_evidence() -> None:
    record = {
        "atom_id": "atom",
        "atom_name": "ageoa.example.atom",
        "wrapper_symbol": "atom",
        "source_kind": "generated_ingest",
        "docstring_summary": "Compute atom output.",
        "has_references": False,
        "source_revision": None,
        "upstream_version": None,
        "has_parity_tests": False,
        "skeleton": False,
        "structural_findings": [],
        "structural_status": "pass",
        "has_docstring": True,
    }
    evidence = {
        "mapping_found": True,
        "upstream_signature": {"parameter_names": ["x"]},
        "findings": [],
        "runtime_probe": {"status": "pass", "findings": ["RUNTIME_PROBE_PASS"]},
        "return_fidelity": {"status": "fail", "findings": ["RETURN_IGNORES_UPSTREAM_VALUE"]},
        "generated_nouns": {"status": "partial", "findings": ["NOUN_UNDOCUMENTED_OUTPUT"]},
    }
    result = score_acceptability(record, evidence)
    assert result["dimension_evidence"]["runtime_status"] == "pass"
    assert result["dimension_evidence"]["semantic_status"] == "fail"
    assert "RETURN_IGNORES_UPSTREAM_VALUE" in result["major_penalties"]
