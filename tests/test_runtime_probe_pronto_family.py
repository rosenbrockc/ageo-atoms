"""Focused runtime-probe coverage for pronto family packets."""

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


def test_runtime_probe_passes_for_pronto_backlash_filter_family() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.pronto.backlash_filter.initializebacklashfilterstate", "initializebacklashfilterstate"),
        ("ageoa.pronto.backlash_filter.updatealphaparameter", "updatealphaparameter"),
        ("ageoa.pronto.backlash_filter.updatecrossingtimemaximum", "updatecrossingtimemaximum"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.pronto.backlash_filter.atoms", wrapper_symbol)


def test_runtime_probe_passes_for_pronto_blip_filter_family() -> None:
    for atom_name, wrapper_symbol in [
        ("ageoa.pronto.blip_filter.bandpass_filter", "bandpass_filter"),
        ("ageoa.pronto.blip_filter.r_peak_detection", "r_peak_detection"),
        ("ageoa.pronto.blip_filter.peak_correction", "peak_correction"),
        ("ageoa.pronto.blip_filter.template_extraction", "template_extraction"),
        ("ageoa.pronto.blip_filter.heart_rate_computation", "heart_rate_computation"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.pronto.blip_filter.atoms", wrapper_symbol)


def test_runtime_probe_passes_for_pronto_state_readout_usage_equivalent_wrappers() -> None:
    for atom_name, module_path, wrapper_symbol in [
        ("ageoa.pronto.foot_contact.mode_snapshot_readout", "ageoa.pronto.foot_contact.atoms", "mode_snapshot_readout"),
        ("ageoa.pronto.leg_odometer.velocitystatereadout", "ageoa.pronto.leg_odometer.atoms", "velocitystatereadout"),
        ("ageoa.pronto.leg_odometer.posequeryaccessors", "ageoa.pronto.leg_odometer.atoms", "posequeryaccessors"),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_pronto_dynamic_stance_estimator_family() -> None:
    for atom_name, module_path, wrapper_symbol in [
        ("ageoa.pronto.dynamic_stance_estimator.initializefilter", "ageoa.pronto.dynamic_stance_estimator.atoms", "initializefilter"),
        ("ageoa.pronto.dynamic_stance_estimator.predictstep", "ageoa.pronto.dynamic_stance_estimator.atoms", "predictstep"),
        ("ageoa.pronto.dynamic_stance_estimator.updatestep", "ageoa.pronto.dynamic_stance_estimator.atoms", "updatestep"),
        ("ageoa.pronto.dynamic_stance_estimator.querystance", "ageoa.pronto.dynamic_stance_estimator.atoms", "querystance"),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_pronto_dynamic_stance_estimator_d12_family() -> None:
    for atom_name, module_path, wrapper_symbol in [
        ("ageoa.pronto.dynamic_stance_estimator_d12.stancestateinit", "ageoa.pronto.dynamic_stance_estimator_d12.atoms", "stancestateinit"),
        ("ageoa.pronto.dynamic_stance_estimator_d12.stanceestimation", "ageoa.pronto.dynamic_stance_estimator_d12.atoms", "stanceestimation"),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_pronto_state_estimator_update() -> None:
    _assert_probe_passes(
        "ageoa.pronto.state_estimator.update_state_estimate",
        "ageoa.pronto.state_estimator",
        "update_state_estimate",
    )
