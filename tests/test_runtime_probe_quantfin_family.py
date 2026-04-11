"""Focused runtime-probe coverage for quantfin family packets."""

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


def test_runtime_probe_passes_for_quantfin_quick_sim_anti() -> None:
    _assert_probe_passes(
        "ageoa.quantfin.montecarlo.quick_sim_anti",
        "ageoa.quantfin.montecarlo",
        "quick_sim_anti",
    )


def test_runtime_probe_passes_for_quantfin_local_vol_d12_family() -> None:
    for atom_name, module_path, wrapper_symbol in [
        ("ageoa.quantfin.local_vol_d12.allfort", "ageoa.quantfin.local_vol_d12.atoms", "allfort"),
        ("ageoa.quantfin.local_vol_d12.var", "ageoa.quantfin.local_vol_d12.atoms", "var"),
        ("ageoa.quantfin.local_vol_d12.localvol", "ageoa.quantfin.local_vol_d12.atoms", "localvol"),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_quantfin_rng_skip_d12_family() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.quantfin.rng_skip_d12.addmod64", "addmod64"),
        ("ageoa.quantfin.rng_skip_d12.randomword32", "randomword32"),
        ("ageoa.quantfin.rng_skip_d12.randomword64", "randomword64"),
        ("ageoa.quantfin.rng_skip_d12.split", "split"),
        ("ageoa.quantfin.rng_skip_d12.next", "next"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.quantfin.rng_skip_d12.atoms", wrapper_symbol)


def test_runtime_probe_passes_for_quantfin_tdma_solver_d12_family() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.quantfin.tdma_solver_d12.tdmasolver", "tdmasolver"),
        ("ageoa.quantfin.tdma_solver_d12.cotraversevec", "cotraversevec"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.quantfin.tdma_solver_d12.atoms", wrapper_symbol)


def test_runtime_probe_passes_for_quantfin_monte_carlo_anti_d12_helpers() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.quantfin.monte_carlo_anti_d12.avg", "avg"),
        ("ageoa.quantfin.monte_carlo_anti_d12.maxstep", "maxstep"),
        ("ageoa.quantfin.monte_carlo_anti_d12.insertcf", "insertcf"),
        ("ageoa.quantfin.monte_carlo_anti_d12.insertcflist", "insertcflist"),
        ("ageoa.quantfin.monte_carlo_anti_d12.runsimulation", "runsimulation"),
        ("ageoa.quantfin.monte_carlo_anti_d12.runsimulationanti", "runsimulationanti"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.quantfin.monte_carlo_anti_d12.atoms", wrapper_symbol)
