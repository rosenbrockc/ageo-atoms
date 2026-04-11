"""Focused runtime-probe coverage for quant_engine family packets."""

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


def test_runtime_probe_passes_for_quant_engine_ofi_and_execution_helpers() -> None:
    _assert_probe_passes(
        "ageoa.quant_engine.calculate_ofi",
        "ageoa.quant_engine.atoms",
        "calculate_ofi",
    )
    _assert_probe_passes(
        "ageoa.quant_engine.execute_vwap",
        "ageoa.quant_engine.atoms",
        "execute_vwap",
    )
    _assert_probe_passes(
        "ageoa.quant_engine.execute_pov",
        "ageoa.quant_engine.atoms",
        "execute_pov",
    )
    _assert_probe_passes(
        "ageoa.quant_engine.execute_passive",
        "ageoa.quant_engine.atoms",
        "execute_passive",
    )
