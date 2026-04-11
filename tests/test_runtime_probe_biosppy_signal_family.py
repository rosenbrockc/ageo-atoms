"""Focused runtime-probe coverage for biosppy signal family packets."""

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


def test_runtime_probe_passes_for_biosppy_ppg_detectors() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ppg_detectors.detect_signal_onsets_elgendi2013", "detect_signal_onsets_elgendi2013"),
        ("ageoa.biosppy.ppg_detectors.detectonsetevents", "detectonsetevents"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.biosppy.ppg_detectors", symbol)


def test_runtime_probe_passes_for_biosppy_abp() -> None:
    _assert_probe_passes(
        "ageoa.biosppy.abp.audio_onset_detection",
        "ageoa.biosppy.abp",
        "audio_onset_detection",
    )


def test_runtime_probe_passes_for_biosppy_ecg_detectors_threshold_based_asi() -> None:
    _assert_probe_passes(
        "ageoa.biosppy.ecg_detectors.thresholdbasedsignalsegmentation",
        "ageoa.biosppy.ecg_detectors",
        "thresholdbasedsignalsegmentation",
    )


def test_runtime_probe_passes_for_biosppy_eda() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.eda.gamboa_segmenter", "gamboa_segmenter"),
        ("ageoa.biosppy.eda.eda_feature_extraction", "eda_feature_extraction"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.biosppy.eda", symbol)


def test_runtime_probe_passes_for_biosppy_emg_detectors() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.emg_detectors.detect_onsets_with_rest_aware_thresholds", "detect_onsets_with_rest_aware_thresholds"),
        ("ageoa.biosppy.emg_detectors.bonato_onset_detection", "bonato_onset_detection"),
        ("ageoa.biosppy.emg_detectors.threshold_based_onset_detection", "threshold_based_onset_detection"),
        ("ageoa.biosppy.emg_detectors.solnik_onset_detect", "solnik_onset_detect"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.biosppy.emg_detectors", symbol)


def test_runtime_probe_passes_for_biosppy_pcg() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.pcg.shannon_energy", "shannon_energy"),
        ("ageoa.biosppy.pcg.pcg_segmentation", "pcg_segmentation"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.biosppy.pcg", symbol)


def test_runtime_probe_passes_for_biosppy_online_filter_variants() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.biosppy.online_filter.filterstateinit", "ageoa.biosppy.online_filter.atoms", "filterstateinit"),
        ("ageoa.biosppy.online_filter.filterstep", "ageoa.biosppy.online_filter.atoms", "filterstep"),
        ("ageoa.biosppy.online_filter_codex.filterstateinit", "ageoa.biosppy.online_filter_codex.atoms", "filterstateinit"),
        ("ageoa.biosppy.online_filter_codex.filterstep", "ageoa.biosppy.online_filter_codex.atoms", "filterstep"),
        ("ageoa.biosppy.online_filter_v2.filterstateinit", "ageoa.biosppy.online_filter_v2.atoms", "filterstateinit"),
        ("ageoa.biosppy.online_filter_v2.filterstep", "ageoa.biosppy.online_filter_v2.atoms", "filterstep"),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_biosppy_svm_proc_family() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.svm_proc.get_auth_rates", "get_auth_rates"),
        ("ageoa.biosppy.svm_proc.get_id_rates", "get_id_rates"),
        ("ageoa.biosppy.svm_proc.get_subject_results", "get_subject_results"),
        ("ageoa.biosppy.svm_proc.assess_classification", "assess_classification"),
        ("ageoa.biosppy.svm_proc.assess_runs", "assess_runs"),
        ("ageoa.biosppy.svm_proc.combination", "combination"),
        ("ageoa.biosppy.svm_proc.majority_rule", "majority_rule"),
        ("ageoa.biosppy.svm_proc.cross_validation", "cross_validation"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.biosppy.svm_proc.atoms", symbol)


def test_runtime_probe_passes_for_biosppy_hamilton_detectors_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg_detectors.hamilton_segmentation", "hamilton_segmentation"),
        ("ageoa.biosppy.ecg_detectors.hamilton_segmenter", "hamilton_segmenter"),
    ]:
        _assert_probe_passes(atom_name, "ageoa.biosppy.ecg_detectors", symbol)


def test_runtime_probe_passes_for_biosppy_zz2018_wrappers_usage_equivalent() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.biosppy.ecg_zz2018.calculatecompositesqi_zz2018", "ageoa.biosppy.ecg_zz2018.atoms", "calculatecompositesqi_zz2018"),
        ("ageoa.biosppy.ecg_zz2018.calculatebeatagreementsqi", "ageoa.biosppy.ecg_zz2018.atoms", "calculatebeatagreementsqi"),
        ("ageoa.biosppy.ecg_zz2018.calculatefrequencypowersqi", "ageoa.biosppy.ecg_zz2018.atoms", "calculatefrequencypowersqi"),
        ("ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi", "ageoa.biosppy.ecg_zz2018_d12.atoms", "assemblezz2018sqi"),
        ("ageoa.biosppy.ecg_zz2018_d12.computebeatagreementsqi", "ageoa.biosppy.ecg_zz2018_d12.atoms", "computebeatagreementsqi"),
        ("ageoa.biosppy.ecg_zz2018_d12.computefrequencysqi", "ageoa.biosppy.ecg_zz2018_d12.atoms", "computefrequencysqi"),
        ("ageoa.biosppy.ecg_zz2018_d12.computekurtosissqi", "ageoa.biosppy.ecg_zz2018_d12.atoms", "computekurtosissqi"),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)
