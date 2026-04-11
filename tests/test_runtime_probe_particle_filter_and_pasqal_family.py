"""Focused runtime-probe coverage for particle_filter_and_pasqal family packets."""

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


def _assert_usage_equivalent_probe_passes(
    atom_name: str, module_import_path: str, wrapper_symbol: str
) -> None:
    probe = runtime_probes.build_runtime_probe(_record(atom_name, module_import_path, wrapper_symbol))
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_particle_filter_basic_helpers() -> None:
    for atom_name, symbol in [
        ("ageoa.particle_filters.basic.filter_step_preparation_and_dispatch", "filter_step_preparation_and_dispatch"),
        ("ageoa.particle_filters.basic.particle_propagation_kernel", "particle_propagation_kernel"),
        ("ageoa.particle_filters.basic.likelihood_reweight_kernel", "likelihood_reweight_kernel"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.particle_filters.basic.atoms", symbol)


def test_runtime_probe_passes_for_particle_filter_resample_projection() -> None:
    _assert_usage_equivalent_probe_passes(
        "ageoa.particle_filters.basic.resample_and_belief_projection",
        "ageoa.particle_filters.basic.atoms",
        "resample_and_belief_projection",
    )


def test_runtime_probe_passes_for_pasqal_sub_graph_embedder() -> None:
    _assert_usage_equivalent_probe_passes(
        "ageoa.pasqal.docking.sub_graph_embedder",
        "ageoa.pasqal.docking",
        "sub_graph_embedder",
    )


def test_runtime_probe_passes_for_pasqal_quantum_mwis_solver() -> None:
    _assert_probe_passes(
        "ageoa.pasqal.docking.quantum_mwis_solver",
        "ageoa.pasqal.docking",
        "quantum_mwis_solver",
    )
