"""Focused runtime-probe coverage for mcmc_foundational family packets."""

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


def test_runtime_probe_passes_for_kthohr_mcmc_builder_family() -> None:
    for atom_name, module_path, symbol in [
        (
            "ageoa.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel",
            "ageoa.mcmc_foundational.kthohr_mcmc.aees.atoms",
            "metropolishastingstransitionkernel",
        ),
        (
            "ageoa.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle",
            "ageoa.mcmc_foundational.kthohr_mcmc.aees.atoms",
            "targetlogkerneloracle",
        ),
        (
            "ageoa.mcmc_foundational.kthohr_mcmc.hmc.buildhmckernelfromlogdensityoracle",
            "ageoa.mcmc_foundational.kthohr_mcmc.hmc",
            "buildhmckernelfromlogdensityoracle",
        ),
        (
            "ageoa.mcmc_foundational.kthohr_mcmc.nuts.nuts_recursive_tree_build",
            "ageoa.mcmc_foundational.kthohr_mcmc.nuts",
            "nuts_recursive_tree_build",
        ),
        (
            "ageoa.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel",
            "ageoa.mcmc_foundational.kthohr_mcmc.rmhmc",
            "buildrmhmctransitionkernel",
        ),
        (
            "ageoa.mcmc_foundational.kthohr_mcmc.rwmh.constructrandomwalkmetropoliskernel",
            "ageoa.mcmc_foundational.kthohr_mcmc.rwmh",
            "constructrandomwalkmetropoliskernel",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_mini_mcmc_hmc_llm_and_nuts_llm_builders() -> None:
    for atom_name, module_path, symbol in [
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializehmckernelstate",
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.atoms",
            "initializehmckernelstate",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializesamplerrng",
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.atoms",
            "initializesamplerrng",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.hamiltoniantransitionkernel",
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.atoms",
            "hamiltoniantransitionkernel",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.collectposteriorchain",
            "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.atoms",
            "collectposteriorchain",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.initializenutsstate",
            "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.atoms",
            "initializenutsstate",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions",
            "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.atoms",
            "runnutstransitions",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_mini_mcmc_kernel_family() -> None:
    for atom_name, module_path, symbol in [
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc.initializehmcstate",
            "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms",
            "initializehmcstate",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc.leapfrogproposalkernel",
            "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms",
            "leapfrogproposalkernel",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition",
            "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms",
            "metropolishmctransition",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.hmc.runsamplingloop",
            "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms",
            "runsamplingloop",
        ),
        (
            "ageoa.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build",
            "ageoa.mcmc_foundational.mini_mcmc.nuts",
            "nuts_recursive_tree_build",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_advancedhmc_trajectory_family() -> None:
    for atom_name, module_path, symbol in [
        (
            "ageoa.mcmc_foundational.advancedhmc.trajectory.buildnutstree",
            "ageoa.mcmc_foundational.advancedhmc.trajectory.atoms",
            "buildnutstree",
        ),
        (
            "ageoa.mcmc_foundational.advancedhmc.trajectory.nutstransitionkernel",
            "ageoa.mcmc_foundational.advancedhmc.trajectory.atoms",
            "nutstransitionkernel",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)
