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
    record = _record_for(manifest, "biosppy/ecg_detectors:thresholdbasedsignalsegmentation")
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


def test_extract_patches_manifest_is_not_marked_stateful_from_random_state_name() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "sklearn/images:extract_patches_2d")
    assert record["stateful_kind"] == "none"
    assert "stateful_api" not in record["risk_reasons"]


def test_extract_patches_mapping_resolves_to_importable_sklearn_signature() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "sklearn/images:extract_patches_2d")
    evidence = build_signature_evidence(record)
    assert evidence["mapping_found"] is True
    assert evidence["upstream_signature_source"] == "inspect"
    assert evidence["upstream_signature"]["parameter_names"] == [
        "image",
        "patch_size",
        "max_patches",
        "random_state",
    ]


def test_numpy_random_manifest_records_installed_upstream_version() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "numpy/random:default_rng")
    assert record["upstream_version"]
    assert any(source.get("kind") == "installed_package" for source in record["authoritative_sources"])


def test_numpy_random_mapping_resolves_to_imported_signature() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "numpy/random:uniform")
    evidence = build_signature_evidence(record)
    assert evidence["mapping_found"] is True
    assert evidence["upstream_mapping"]["module"] == "numpy.random"
    assert evidence["upstream_mapping"]["function"] == "uniform"


def test_fasta_dataset_manifest_is_not_marked_ffi_from_sort_method_name() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "mint/fasta_dataset:dataset_state_initialization")
    assert record["ffi"] is False


def test_fasta_dataset_state_adapter_is_not_treated_as_invented_parameter() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "mint/fasta_dataset:dataset_length_query")
    evidence = build_signature_evidence(record)
    assert "FIDELITY_SIGNATURE_INVENTED_PARAMETER" not in evidence["findings"]
    assert "FIDELITY_REQUIREDNESS_MISMATCH" not in evidence["findings"]


def test_online_filter_state_adapter_is_not_treated_as_invented_parameter() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "biosppy/online_filter:filterstep")
    evidence = build_signature_evidence(record)
    assert "FIDELITY_SIGNATURE_INVENTED_PARAMETER" not in evidence["findings"]
    assert "FIDELITY_REQUIREDNESS_MISMATCH" not in evidence["findings"]
