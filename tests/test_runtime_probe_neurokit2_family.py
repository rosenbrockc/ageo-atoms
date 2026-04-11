"""Focused runtime-probe coverage for neurokit2 family packets."""

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


def test_runtime_probe_passes_for_neurokit2_wrappers_usage_equivalent() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.neurokit2.averageqrstemplate", "averageqrstemplate"),
        ("ageoa.neurokit2.zhao2018hrvanalysis", "zhao2018hrvanalysis"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.neurokit2.atoms", wrapper_symbol)
