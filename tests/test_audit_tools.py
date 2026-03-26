from __future__ import annotations

from auditlib.acceptability import score_acceptability
from auditlib.fidelity import build_signature_evidence
from auditlib.inventory import build_manifest


def _record_for(manifest: dict, atom_key: str) -> dict:
    for record in manifest["atoms"]:
        if record["atom_key"] == atom_key:
            return record
    raise AssertionError(f"Missing manifest record for {atom_key}")


def test_build_manifest_discovers_known_registered_atoms() -> None:
    manifest = build_manifest()
    assert "metadata" in manifest
    assert "summary" in manifest
    record = _record_for(manifest, "algorithms/graph:bfs")
    assert record["atom_name"] == "ageoa.algorithms.graph.bfs"
    assert record["module_import_path"] == "ageoa.algorithms.graph"
    assert record["has_witnesses"] is True
    assert record["stateful_kind"] == "none"
    assert isinstance(record["authoritative_sources"], list)
    assert isinstance(record["risk_reasons"], list)


def test_signature_fidelity_uses_vendored_source_when_mapped() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "biosppy/ecg_asi:thresholdbasedsignalsegmentation")
    evidence = build_signature_evidence(record)
    assert evidence["mapping_found"] is True
    assert evidence["upstream_signature_source"] == "vendored_ast"
    assert evidence["upstream_signature"]["parameter_names"] == ["signal", "sampling_rate", "Pth"]


def test_acceptability_caps_unmapped_atoms_below_trusted_range() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "algorithms/graph:bfs")
    evidence = build_signature_evidence(record)
    result = score_acceptability(record, evidence)
    assert result["acceptability_score"] <= 59
    assert result["max_reviewable_verdict"] == "acceptable_with_limits"
