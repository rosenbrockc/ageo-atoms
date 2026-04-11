"""Focused runtime-probe coverage for advancedvi_and_iqe family packets."""

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
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_advancedvi_core_wrappers() -> None:
    _assert_probe_passes(
        "ageoa.advancedvi.core.evaluate_log_probability_density",
        "ageoa.advancedvi.core",
        "evaluate_log_probability_density",
    )
    _assert_probe_passes(
        "ageoa.advancedvi.core.gradient_oracle_evaluation",
        "ageoa.advancedvi.core",
        "gradient_oracle_evaluation",
    )


def test_runtime_probe_passes_for_institutional_quant_engine_wrappers() -> None:
    _assert_probe_passes(
        "ageoa.institutional_quant_engine.hawkes_process.hawkesprocesssimulator",
        "ageoa.institutional_quant_engine.hawkes_process",
        "hawkesprocesssimulator",
    )
    _assert_probe_passes(
        "ageoa.institutional_quant_engine.heston_model.hestonpathsampler",
        "ageoa.institutional_quant_engine.heston_model",
        "hestonpathsampler",
    )
    _assert_probe_passes(
        "ageoa.institutional_quant_engine.avellaneda_stoikov.computeinventoryadjustedquotes",
        "ageoa.institutional_quant_engine.avellaneda_stoikov.atoms",
        "computeinventoryadjustedquotes",
    )
