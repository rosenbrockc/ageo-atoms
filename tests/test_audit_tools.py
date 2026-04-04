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


def test_rust_runsamplingloop_mapping_resolves_to_vendored_rust_signature() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "mcmc_foundational/mini_mcmc/hmc:runsamplingloop")
    evidence = build_signature_evidence(record)
    assert evidence["mapping_found"] is True
    assert evidence["upstream_signature_source"] == "vendored_rust"
    assert evidence["upstream_signature"]["parameter_names"] == ["chain", "n_collect", "n_discard"]
    assert "FIDELITY_SIGNATURE_INVENTED_PARAMETER" not in evidence["findings"]
    assert "FIDELITY_SIGNATURE_MISSING_REQUIRED" not in evidence["findings"]


def test_rust_bicycle_kinematic_mapping_resolves_trait_impl_signature() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "rust_robotics/bicycle_kinematic:evaluateandinvertdynamics")
    evidence = build_signature_evidence(record)
    assert evidence["mapping_found"] is True
    assert evidence["upstream_signature_source"] == "vendored_rust"
    assert evidence["upstream_signature"]["parameter_names"] == ["x", "u", "_t"]


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


def test_greedy_mapping_context_adapter_is_not_treated_as_invented_parameter() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "molecular_docking/greedy_mapping:initializefrontierfromstartnode")
    evidence = build_signature_evidence(record)
    assert "FIDELITY_SIGNATURE_INVENTED_PARAMETER" not in evidence["findings"]
    assert "FIDELITY_REQUIREDNESS_MISMATCH" not in evidence["findings"]


def test_greedy_mapping_pipeline_decomposition_outputs_are_not_treated_as_invented_parameters() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "molecular_docking/greedy_mapping:rungreedymappingpipeline")
    evidence = build_signature_evidence(record)
    assert "FIDELITY_SIGNATURE_INVENTED_PARAMETER" not in evidence["findings"]
    assert "FIDELITY_REQUIREDNESS_MISMATCH" not in evidence["findings"]


def test_minimize_bandwidth_loop_stage_adapters_do_not_trigger_signature_findings() -> None:
    manifest = build_manifest()
    for atom_key in [
        "molecular_docking/minimize_bandwidth:propose_greedy_permutation_step",
        "molecular_docking/minimize_bandwidth:update_state_with_improvement_criterion",
        "molecular_docking/minimize_bandwidth:extract_final_permutation",
    ]:
        evidence = build_signature_evidence(_record_for(manifest, atom_key))
        assert "FIDELITY_SIGNATURE_INVENTED_PARAMETER" not in evidence["findings"]
        assert "FIDELITY_SIGNATURE_MISSING_REQUIRED" not in evidence["findings"]
        assert "FIDELITY_SIGNATURE_ORDER_MISMATCH" not in evidence["findings"]


def test_loopy_bp_state_in_adapter_is_not_treated_as_invented_parameter() -> None:
    manifest = build_manifest()
    record = _record_for(manifest, "belief_propagation/loopy_bp:run_loopy_message_passing_and_belief_query")
    evidence = build_signature_evidence(record)
    assert "FIDELITY_SIGNATURE_INVENTED_PARAMETER" not in evidence["findings"]
    assert "FIDELITY_REQUIREDNESS_MISMATCH" not in evidence["findings"]
