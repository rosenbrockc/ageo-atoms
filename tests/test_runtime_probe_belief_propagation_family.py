"""Focused runtime-probe coverage for belief_propagation family packets."""

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


def test_runtime_probe_passes_for_belief_propagation_loopy_bp_family() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.belief_propagation.loopy_bp.initialize_message_passing_state", "initialize_message_passing_state"),
        ("ageoa.belief_propagation.loopy_bp.run_loopy_message_passing_and_belief_query", "run_loopy_message_passing_and_belief_query"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.belief_propagation.loopy_bp.atoms", wrapper_symbol)
