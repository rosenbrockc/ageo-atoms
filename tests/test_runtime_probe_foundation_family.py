"""Focused runtime-probe coverage for foundation-style family packets."""

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


def test_runtime_probe_passes_for_e2e_ppg_family() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.e2e_ppg.kazemi_peak_detection", "ageoa.e2e_ppg.atoms", "kazemi_peak_detection"),
        ("ageoa.e2e_ppg.ppg_reconstruction", "ageoa.e2e_ppg.atoms", "ppg_reconstruction"),
        ("ageoa.e2e_ppg.ppg_sqa", "ageoa.e2e_ppg.atoms", "ppg_sqa"),
        ("ageoa.e2e_ppg.template_matching.templatefeaturecomputation", "ageoa.e2e_ppg.template_matching", "templatefeaturecomputation"),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_e2e_ppg_windowed_reconstruction() -> None:
    _assert_probe_passes(
        "ageoa.e2e_ppg.reconstruction.windowed_signal_reconstruction",
        "ageoa.e2e_ppg.reconstruction.atoms",
        "windowed_signal_reconstruction",
    )


def test_runtime_probe_passes_for_e2e_ppg_kazemi_wrapper_d12_helpers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.e2e_ppg.kazemi_wrapper_d12.normalizesignal", "ageoa.e2e_ppg.kazemi_wrapper_d12.atoms", "normalizesignal"),
        ("ageoa.e2e_ppg.kazemi_wrapper_d12.wrapperevaluate", "ageoa.e2e_ppg.kazemi_wrapper_d12.atoms", "wrapperevaluate"),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_mint_axial_attention_helpers() -> None:
    for atom_name, symbol in [
        ("ageoa.mint.axial_attention.rowselfattention", "rowselfattention"),
        ("ageoa.mint.axial_attention.row_self_attention", "row_self_attention"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.mint.axial_attention", symbol)


def test_runtime_probe_passes_for_mint_top_level_attention_atoms() -> None:
    for atom_name, symbol in [
        ("ageoa.mint.axial_attention", "axial_attention"),
        ("ageoa.mint.rotary_positional_embeddings", "rotary_positional_embeddings"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.mint.atoms", symbol)


def test_runtime_probe_passes_for_alphafold_atoms() -> None:
    for atom_name, symbol in [
        ("ageoa.alphafold.invariant_point_attention", "invariant_point_attention"),
        ("ageoa.alphafold.equivariant_frame_update", "equivariant_frame_update"),
        ("ageoa.alphafold.coordinate_reconstruction", "coordinate_reconstruction"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.alphafold.atoms", symbol)
