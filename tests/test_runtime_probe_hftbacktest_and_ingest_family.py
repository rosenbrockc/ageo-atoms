"""Focused runtime-probe coverage for hftbacktest_and_ingest family packets."""

from __future__ import annotations

from auditlib import runtime_probes


def _record(atom_name: str, module_import_path: str, wrapper_symbol: str) -> dict[str, object]:
    return {
        "atom_id": f"{atom_name}@ageoa/example.py:1",
        "atom_name": atom_name,
        "module_import_path": module_import_path,
        "module_path": "ageoa/example.py",
        "wrapper_symbol": wrapper_symbol,
        "wrapper_line": 1,
        "skeleton": False,
    }


def _assert_probe_passes(atom_name: str, module_import_path: str, wrapper_symbol: str) -> None:
    probe = runtime_probes.build_runtime_probe(_record(atom_name, module_import_path, wrapper_symbol))
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_hftbacktest_update_glft_coefficients() -> None:
    _assert_probe_passes(
        "ageoa.hftbacktest.update_glft_coefficients",
        "ageoa.hftbacktest.atoms",
        "update_glft_coefficients",
    )


def test_runtime_probe_passes_for_mint_encoding_dist_mat_family() -> None:
    _assert_probe_passes(
        "ageoa.mint.encoding_dist_mat.encodedistancematrix",
        "ageoa.mint.encoding_dist_mat",
        "encodedistancematrix",
    )


def test_runtime_probe_passes_for_mint_fasta_dataset_family() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.mint.fasta_dataset.dataset_state_initialization", "dataset_state_initialization"),
        ("ageoa.mint.fasta_dataset.dataset_length_query", "dataset_length_query"),
        ("ageoa.mint.fasta_dataset.dataset_item_retrieval", "dataset_item_retrieval"),
        ("ageoa.mint.fasta_dataset.token_budget_batch_planning", "token_budget_batch_planning"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.mint.fasta_dataset.atoms", wrapper_symbol)


def test_runtime_probe_passes_for_mint_incremental_attention_configuration() -> None:
    _assert_probe_passes(
        "ageoa.mint.incremental_attention.enable_incremental_state_configuration",
        "ageoa.mint.incremental_attention",
        "enable_incremental_state_configuration",
    )
