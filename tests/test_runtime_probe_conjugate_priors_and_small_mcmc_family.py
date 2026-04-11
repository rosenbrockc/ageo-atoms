"""Focused runtime-probe coverage for conjugate_priors_and_small_mcmc family packets."""

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


def test_runtime_probe_passes_for_conjugate_priors_wrappers() -> None:
    _assert_probe_passes(
        "ageoa.conjugate_priors.beta_binom.posterior_randmodel",
        "ageoa.conjugate_priors.beta_binom.atoms",
        "posterior_randmodel",
    )
    _assert_probe_passes(
        "ageoa.conjugate_priors.normal.normal_gamma_posterior_update",
        "ageoa.conjugate_priors.normal",
        "normal_gamma_posterior_update",
    )


def test_runtime_probe_passes_for_small_kthohr_mcmc_helpers() -> None:
    _assert_probe_passes(
        "ageoa.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel",
        "ageoa.mcmc_foundational.kthohr_mcmc.de",
        "build_de_transition_kernel",
    )
    _assert_probe_passes(
        "ageoa.mcmc_foundational.kthohr_mcmc.mala.mala_proposal_adjustment",
        "ageoa.mcmc_foundational.kthohr_mcmc.mala",
        "mala_proposal_adjustment",
    )
