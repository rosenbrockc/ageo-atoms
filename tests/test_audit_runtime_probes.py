from __future__ import annotations

from types import SimpleNamespace

from auditlib import runtime_probes


def _record(atom_name: str, module_import_path: str = "ageoa.algorithms.search", wrapper_symbol: str = "binary_search") -> dict:
    return {
        "atom_id": f"{atom_name}@ageoa/example.py:1",
        "atom_name": atom_name,
        "module_import_path": module_import_path,
        "module_path": "ageoa/example.py",
        "wrapper_symbol": wrapper_symbol,
        "wrapper_line": 1,
        "skeleton": False,
    }


def test_runtime_probe_passes_for_safe_real_atom() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.algorithms.search.binary_search", "ageoa.algorithms.search", "binary_search")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_skips_unsupported_atom() -> None:
    probe = runtime_probes.build_runtime_probe(_record("ageoa.example.unsupported", "ageoa.example", "atom"))
    assert probe["status"] == "not_applicable"
    assert probe["skip_reason"] == "unsupported_scope"
    assert probe["findings"] == ["RUNTIME_PROBE_SKIPPED"]


def test_runtime_probe_passes_for_numpy_fft() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.numpy.fft.fft", "ageoa.numpy.fft", "fft")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_sparse_graph_laplacian() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.scipy.sparse_graph.graph_laplacian", "ageoa.scipy.sparse_graph", "graph_laplacian")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_sklearn_image_grid_to_graph() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.sklearn.images.grid_to_graph", "ageoa.sklearn.images.atoms", "grid_to_graph")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_numpy_fft_v2() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.numpy.fft_v2.forwardmultidimensionalfft",
            "ageoa.numpy.fft_v2.atoms",
            "forwardmultidimensionalfft",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_sorting_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.algorithms.sorting.merge_sort", "ageoa.algorithms.sorting", "merge_sort")
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_passes_for_advancedvi_log_density() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.advancedvi.core.evaluate_log_probability_density",
            "ageoa.advancedvi.core",
            "evaluate_log_probability_density",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_hawkes_process_simulator() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.institutional_quant_engine.hawkes_process.hawkesprocesssimulator",
            "ageoa.institutional_quant_engine.hawkes_process",
            "hawkesprocesssimulator",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_heston_path_sampler() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.institutional_quant_engine.heston_model.hestonpathsampler",
            "ageoa.institutional_quant_engine.heston_model",
            "hestonpathsampler",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_particle_filter_resample_projection() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.particle_filters.basic.resample_and_belief_projection",
            "ageoa.particle_filters.basic.atoms",
            "resample_and_belief_projection",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pasqal_sub_graph_embedder() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.pasqal.docking.sub_graph_embedder",
            "ageoa.pasqal.docking",
            "sub_graph_embedder",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_scipy_optimize_v2_shgo() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.scipy.optimize_v2.shgoglobaloptimization",
            "ageoa.scipy.optimize_v2.atoms",
            "shgoglobaloptimization",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_ppg_detectors() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ppg_detectors.detect_signal_onsets_elgendi2013", "detect_signal_onsets_elgendi2013"),
        ("ageoa.biosppy.ppg_detectors.detectonsetevents", "detectonsetevents"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ppg_detectors", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_abp() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.biosppy.abp.audio_onset_detection",
            "ageoa.biosppy.abp",
            "audio_onset_detection",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_ecg_module_wrappers() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg.bandpass_filter", "bandpass_filter"),
        ("ageoa.biosppy.ecg.r_peak_detection", "r_peak_detection"),
        ("ageoa.biosppy.ecg.peak_correction", "peak_correction"),
        ("ageoa.biosppy.ecg.template_extraction", "template_extraction"),
        ("ageoa.biosppy.ecg.heart_rate_computation", "heart_rate_computation"),
        ("ageoa.biosppy.ecg.ssf_segmenter", "ssf_segmenter"),
        ("ageoa.biosppy.ecg.christov_segmenter", "christov_segmenter"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ecg", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_threshold_based_asi() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.biosppy.ecg_detectors.thresholdbasedsignalsegmentation",
            "ageoa.biosppy.ecg_detectors",
            "thresholdbasedsignalsegmentation",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_eda() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.eda.gamboa_segmenter", "gamboa_segmenter"),
        ("ageoa.biosppy.eda.eda_feature_extraction", "eda_feature_extraction"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.eda", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_emg_detectors() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.emg_detectors.detect_onsets_with_rest_aware_thresholds", "detect_onsets_with_rest_aware_thresholds"),
        ("ageoa.biosppy.emg_detectors.bonato_onset_detection", "bonato_onset_detection"),
        ("ageoa.biosppy.emg_detectors.threshold_based_onset_detection", "threshold_based_onset_detection"),
        ("ageoa.biosppy.emg_detectors.solnik_onset_detect", "solnik_onset_detect"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.emg_detectors", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_pcg() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.pcg.shannon_energy", "shannon_energy"),
        ("ageoa.biosppy.pcg.pcg_segmentation", "pcg_segmentation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.pcg", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_hftbacktest_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.hftbacktest.update_glft_coefficients", "ageoa.hftbacktest.atoms", "update_glft_coefficients")
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_institutional_quant_engine_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.institutional_quant_engine.market_making_avellaneda",
            "ageoa.institutional_quant_engine.atoms",
            "market_making_avellaneda",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_passes_for_incremental_attention_configuration() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.mint.incremental_attention.enable_incremental_state_configuration",
            "ageoa.mint.incremental_attention",
            "enable_incremental_state_configuration",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_greedy_subgraph() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.greedy_subgraph.greedy_maximum_subgraph",
            "ageoa.molecular_docking.greedy_subgraph",
            "greedy_maximum_subgraph",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_quantum_mwis_solver_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.quantum_mwis_solver",
            "ageoa.molecular_docking.atoms",
            "quantum_mwis_solver",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_add_quantum_link_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.add_quantum_link.addquantumlink",
            "ageoa.molecular_docking.add_quantum_link",
            "addquantumlink",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_mwis_to_qubo_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.mwis_sa.to_qubo",
            "ageoa.molecular_docking.mwis_sa.atoms",
            "to_qubo",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_threshold_permutation_enumeration_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.minimize_bandwidth.enumerate_threshold_based_permutations",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "enumerate_threshold_based_permutations",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_minimize_bandwidth_validation_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.minimize_bandwidth.validate_square_matrix_shape", "validate_square_matrix_shape"),
        ("ageoa.molecular_docking.minimize_bandwidth.validate_symmetric_input", "validate_symmetric_input"),
        ("ageoa.molecular_docking.minimize_bandwidth.enforce_threshold_sparsity", "enforce_threshold_sparsity"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.minimize_bandwidth.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_minimize_bandwidth_state_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.minimize_bandwidth.initialize_reduction_state", "initialize_reduction_state"),
        ("ageoa.molecular_docking.minimize_bandwidth.extract_final_permutation", "extract_final_permutation"),
        ("ageoa.molecular_docking.minimize_bandwidth.select_minimum_bandwidth_permutation", "select_minimum_bandwidth_permutation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.minimize_bandwidth.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_minimize_bandwidth_numeric_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.minimize_bandwidth.compute_absolute_weighted_index_distances", "compute_absolute_weighted_index_distances"),
        ("ageoa.molecular_docking.minimize_bandwidth.aggregate_maximum_distance_as_bandwidth", "aggregate_maximum_distance_as_bandwidth"),
        ("ageoa.molecular_docking.minimize_bandwidth.build_sparse_graph_view", "build_sparse_graph_view"),
        ("ageoa.molecular_docking.minimize_bandwidth.compute_symmetric_bandwidth_reducing_order", "compute_symmetric_bandwidth_reducing_order"),
        ("ageoa.molecular_docking.minimize_bandwidth.build_threshold_search_space", "build_threshold_search_space"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.minimize_bandwidth.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_numpy_lexsort_v2_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.numpy.search_sort_v2.lexicographicindirectsort",
            "ageoa.numpy.search_sort_v2.atoms",
            "lexicographicindirectsort",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_biosppy_hamilton_detectors_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg_detectors.hamilton_segmentation", "hamilton_segmentation"),
        ("ageoa.biosppy.ecg_detectors.hamilton_segmenter", "hamilton_segmenter"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ecg_detectors", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_biosppy_zz2018_main_wrappers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg_zz2018.calculatecompositesqi_zz2018", "calculatecompositesqi_zz2018"),
        ("ageoa.biosppy.ecg_zz2018.calculatebeatagreementsqi", "calculatebeatagreementsqi"),
        ("ageoa.biosppy.ecg_zz2018.calculatefrequencypowersqi", "calculatefrequencypowersqi"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ecg_zz2018.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_records_positive_failure(monkeypatch) -> None:
    atom_name = "ageoa.example.fail"
    monkeypatch.setitem(
        runtime_probes.PROBE_PLANS,
        atom_name,
        runtime_probes.ProbePlan(
            positive=runtime_probes.ProbeCase("always fails", lambda func: func(), None),
            negative=runtime_probes.ProbeCase("negative passes", lambda func: (_ for _ in ()).throw(ValueError("bad")), expect_exception=True),
        ),
    )
    monkeypatch.setattr(runtime_probes, "safe_import_module", lambda _: SimpleNamespace(atom=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    probe = runtime_probes.build_runtime_probe(_record(atom_name, "ageoa.example", "atom"))
    assert probe["status"] == "fail"
    assert "RUNTIME_PROBE_FAIL" in probe["findings"]
    assert probe["positive_probe"]["exception_type"] == "RuntimeError"
